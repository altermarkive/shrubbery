import argparse
import gc
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Callable
from warnings import deprecated

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.model_selection import GridSearchCV

from shrubbery.constants import (
    COLUMN_ERA,
    COLUMN_ID,
    RANDOM_SEED,
)
from shrubbery.data.augmentation import override_numerai_era
from shrubbery.data.ingest import (
    download_numerai_files,
    get_feature_set,
    get_training_targets,
    read_parquet_and_unpack,
)
from shrubbery.metrics import submit_diagnostic_predictions
from shrubbery.napi import napi
from shrubbery.observability import logger, silence_false_positive_warnings
from shrubbery.tournament import (
    submit_tournament_predictions,
    update_tournament_submissions,
)
from shrubbery.utilities import load_model, store_model
from shrubbery.validation import (
    cross_validation_to_parallel_coordinates,
    get_best_parameters,
    reformat_cross_validation_result,
)
from shrubbery.workspace import get_workspace_path


@deprecated('Use GridSearchCV with verbose set to 10 instead')
class WandbGridSearchCV(GridSearchCV):
    def __init__(
        self,
        model_name: str,
        estimator: Any,
        param_grid: dict,
        scoring: Callable,
        n_jobs: int | None = None,
        refit: bool = True,
        cv: Any = None,
        verbose: int = 0,
        pre_dispatch: int | str = '2*n_jobs',
        error_score: float = np.nan,
        return_train_score: bool = False,
    ) -> None:
        self.model_name = model_name
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score

    def fit(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict[str, Any]
    ) -> 'WandbGridSearchCV':
        logger.info('Entering time series cross validation loop')
        logger.info(f'Shape of training data - x:{x.shape} y:{y.shape}')
        super().fit(x, y)
        cv_results = self.cv_results_
        cross_validation_to_parallel_coordinates(cv_results, self.model_name)
        cross_validation_result = reformat_cross_validation_result(
            cv_results, self.model_name
        )
        for parameter in self.param_grid.keys():
            get_best_parameters(cross_validation_result, parameter, top_k=1)
        return self.best_estimator_

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.best_estimator_.predict(x)


class NumeraiRunner:
    def __init__(
        self,
        feature_set_name: str,
        retrain: bool,
        estimator: Any,
        numerai_model_id: str,
        version: str,
        notes: str,
        deterministic: bool,
    ) -> None:
        self.feature_set_name = feature_set_name
        self.retrain = retrain
        self.estimator = estimator
        self.numerai_model_id = numerai_model_id
        self.version = version
        self.notes = notes
        self.deterministic = deterministic

    def run(self, config_content: bytes, config_name: str) -> None:
        if self.deterministic:
            # Seeding pins every stochastic component (weight init, dropout,
            # DataLoader shuffling, GAN/denoise noise, unseeded RF bootstrap)
            # to a single draw. That makes runs reproducible but can lock
            # training onto a worse-than-average outcome compared to an
            # unseeded run, so only enable it when you specifically need
            # determinism.
            torch.manual_seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
        silence_false_positive_warnings()
        update_tournament_submissions(self.numerai_model_id)
        wandb.init(dir='/tmp/wandb')
        _save_config_file_to_wandb(config_content, config_name)
        tournament_round = napi.get_current_round()
        logger.info(f'Tournament round: {tournament_round}')
        logger.info(f'Model Name: {self.numerai_model_id}')
        logger.info(f'Notes: {self.notes}')
        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    'numerai_model_id': self.numerai_model_id,
                    'tournament_round': tournament_round,
                }
            )
            wandb.run.notes = self.notes
        download_numerai_files()
        feature_cols = get_feature_set(self.feature_set_name)
        targets = get_training_targets()
        read_columns = [COLUMN_ERA] + feature_cols + targets

        training_data, training_eras = read_parquet_and_unpack(
            'train.parquet', read_columns, feature_cols
        )
        validation_data, validation_eras = read_parquet_and_unpack(
            'validation.parquet', read_columns, feature_cols
        )
        live_data, _ = read_parquet_and_unpack(
            'live.parquet', read_columns, feature_cols
        )
        override_numerai_era(training_eras + validation_eras, live_data)

        # Check for nans and fill nans
        nans_per_col = live_data[feature_cols].isna().sum()
        logger.info('Checking for nans in the tournament data')
        if nans_per_col.any():
            total_rows = live_data.shape[0]
            nans_per_col_count = nans_per_col[nans_per_col > 0]
            logger.info(
                f'Number of nans per column this week: {nans_per_col_count}'
            )
            logger.info(f'Out of {total_rows} total rows')
            logger.info('Filling nans with 0.5')
            live_data.loc[:, feature_cols] = live_data.loc[
                :, feature_cols
            ].fillna(0.5)
        else:
            logger.info('No nans in the features this week!')
        # Load model if present
        model_name = f'model_{self.numerai_model_id}'
        model_file = Path(os.environ['NUMERAI_MODEL_PATH'])
        model = None if self.retrain else load_model(model_file)
        if model is None:
            logger.info(f'Training model: {model_name}')
            self.estimator = self.estimator.fit(
                training_data[[COLUMN_ERA] + feature_cols].to_numpy(),
                training_data[targets].to_numpy(),
            )
            store_model(self.estimator, model_file)
        else:
            self.estimator = model
        gc.collect()

        tournament_data = pd.DataFrame(live_data.index).set_index(COLUMN_ID)
        tournament_data['predictions'] = self.estimator.predict(
            live_data[[COLUMN_ERA] + feature_cols].to_numpy()
        )
        gc.collect()
        submit_tournament_predictions(tournament_data, self.numerai_model_id)

        try:
            diagnostic_data = pd.DataFrame(validation_data.index).set_index(
                COLUMN_ID
            )
            diagnostic_data['predictions'] = self.estimator.predict(
                validation_data[[COLUMN_ERA] + feature_cols].to_numpy()
            )
            gc.collect()
            submit_diagnostic_predictions(
                diagnostic_data, self.numerai_model_id
            )
        except MemoryError:
            traceback.print_exc()

        wandb.finish()


def _save_config_file_to_wandb(
    config_content: bytes, config_name: str
) -> None:
    directory = get_workspace_path()
    config_path = directory / config_name
    with open(config_path, 'wb') as handle:
        handle.write(config_content)
    wandb.save(config_path, base_path=directory)


def config_content(config_path: str) -> bytes:
    return Path(config_path).read_bytes()


def main_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Shrubbery')
    parser.add_argument(
        '--retrain', action='store_true', help='Use this flag to retrain'
    )
    parser.add_argument(
        '--training-era-stride',
        type=int,
        default=1,
        help='Use this argument to downsample eras by given stride',
    )
    parser.add_argument(
        '--debug', action='store_true', help='Start bash shell in-situ'
    )
    arguments = parser.parse_args()
    if arguments.debug:
        subprocess.run('/bin/bash')
        sys.exit()
    return arguments
