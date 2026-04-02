import argparse
import gc
from pathlib import Path
from typing import Any, Callable, Namespace

import hydra
import hydra.utils
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin
from sklearn.model_selection import GridSearchCV

from shrubbery.adversarial_validation import adversarial_downsampling
from shrubbery.constants import (
    COLUMN_ERA,
    COLUMN_ID,
    COLUMN_INDEX_TARGET,
)
from shrubbery.cross_validation import (
    NumeraiTimeSeriesSplitter,
    cross_validation_to_parallel_coordinates,
    get_best_parameters,
    reformat_cross_validation_result,
)
from shrubbery.data.augmentation import override_numerai_era
from shrubbery.data.downsampling import FitDownsamplerBySample
from shrubbery.data.ingest import (
    download_numerai_files,
    get_feature_set,
    get_training_targets,
    lookup_target_index,
    read_parquet_and_unpack,
)
from shrubbery.meta_estimator import NumeraiMetaEstimator
from shrubbery.metrics import submit_diagnostic_predictions
from shrubbery.napi import napi
from shrubbery.observability import logger, silence_false_positive_warnings
from shrubbery.tournament import (
    submit_tournament_predictions,
    update_tournament_submissions,
)
from shrubbery.utilities import PrintableModelMixin, load_model, store_model
from shrubbery.workspace import get_workspace_path


class NumeraiBestGridSearchEstimator(
    BaseEstimator, MetaEstimatorMixin, RegressorMixin, PrintableModelMixin
):
    def __init__(
        self,
        estimator: Any,
        model_name: str,
        drop_era_column: bool,
        downsample_cross_validation: int,
        downsample_full_train: int,
        targets: list[int | str],
        cv: int,
        cv_param_grid: dict,
        cv_scoring: Callable,
        embargo: int,
        neutralization_feature_indices: list[int] | None,
        neutralization_proportion: float,
        neutralization_normalize: bool,
    ) -> None:
        self.estimator = NumeraiMetaEstimator(
            estimator=estimator,
            drop_era_column=drop_era_column,
            target=COLUMN_INDEX_TARGET,
            neutralization_feature_indices=neutralization_feature_indices,
            neutralization_proportion=neutralization_proportion,
            neutralization_normalize=neutralization_normalize,
        )
        self.model_name = model_name
        self.drop_era_column = drop_era_column
        self.downsample_cross_validation = downsample_cross_validation
        self.downsample_full_train = downsample_full_train
        self.targets = targets
        self.cv = cv
        self.cv_param_grid = cv_param_grid
        self.cv_scoring = cv_scoring
        self.embargo = embargo
        self.neutralization_feature_indices = neutralization_feature_indices
        self.neutralization_proportion = neutralization_proportion
        self.neutralization_normalize = neutralization_normalize

    def fit(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict[str, Any]
    ) -> NumeraiMetaEstimator:
        logger.info(f'Shape of training data - x:{x.shape} y:{y.shape}')
        targets = [
            target if isinstance(target, int) else lookup_target_index(target)
            for target in self.targets
        ]
        y = y[:, targets]

        logger.info('Entering time series cross validation loop')
        grid_search_cv = GridSearchCV(
            self.estimator,
            param_grid=self.cv_param_grid | {'target': range(y.shape[1])},
            cv=NumeraiTimeSeriesSplitter(
                cv=self.cv,
                embargo=self.embargo,
            ),
            scoring=self.cv_scoring,
            refit=False,
        )
        FitDownsamplerBySample(
            estimator=grid_search_cv,
            sample_stride=self.downsample_cross_validation,
        ).fit(x, y)

        cv_results = grid_search_cv.cv_results_
        logger.info(
            f'Cross-validation results {self.model_name}: {cv_results}'
        )
        cross_validation_to_parallel_coordinates(cv_results, self.model_name)
        cross_validation_result = reformat_cross_validation_result(
            cv_results, self.model_name
        )
        self.best_target = get_best_parameters(
            cross_validation_result, 'target', top_k=1
        )[0]
        best_model_parameters: dict = {
            parameter: get_best_parameters(
                cross_validation_result, parameter, top_k=1
            )[0]
            for parameter in self.cv_param_grid.keys()
        }
        model_parameters = best_model_parameters | {'target': self.best_target}
        logger.info(f'Model creator arguments: {model_parameters}')
        # Now do a full train
        logger.info('Entering full training section')
        self.estimator.set_params(**model_parameters)
        self.estimator.fit(x, y)
        self.fitted_cv_results_ = grid_search_cv.cv_results_
        return self.estimator

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict(x)


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
        adversarial_downsampling_ratio: float | None,
    ) -> None:
        self.feature_set_name = feature_set_name
        self.retrain = retrain
        self.estimator = estimator
        self.numerai_model_id = numerai_model_id
        self.version = version
        self.notes = notes
        self.adversarial_downsampling_ratio = adversarial_downsampling_ratio

    def run(self, config_content: bytes, config_name: str) -> None:
        silence_false_positive_warnings()
        update_tournament_submissions(self.numerai_model_id)
        wandb.init(dir='/tmp/wandb')
        _save_config_file_to_wandb(config_content, config_name)
        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    'numerai_model_id': self.numerai_model_id,
                    'tournament_round': napi.get_current_round(),
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
        if self.retrain and self.adversarial_downsampling_ratio is not None:
            _, training_data, _ = adversarial_downsampling(
                feature_cols,
                training_data,
                validation_data,
                self.adversarial_downsampling_ratio,
            )

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
        model, version = (
            (None, 'latest')
            if self.retrain
            else load_model(model_name, self.version)
        )
        if model is None:
            logger.info(f'Training model: {model_name}')
            self.estimator = self.estimator.fit(
                training_data[[COLUMN_ERA] + feature_cols].to_numpy(),
                training_data[targets].to_numpy(),
            )
            version = store_model(self.estimator, model_name)
        else:
            self.estimator = model
        version = '' if version == 'latest' else version.replace('v', '')
        model_name = f'{model_name}{version}'
        gc.collect()

        tournament_data = pd.DataFrame(live_data.index).set_index(COLUMN_ID)
        tournament_data['predictions'] = self.estimator.predict(
            live_data[[COLUMN_ERA] + feature_cols].to_numpy()
        )
        gc.collect()
        submit_tournament_predictions(tournament_data, self.numerai_model_id)

        diagnostic_data = pd.DataFrame(validation_data.index).set_index(
            COLUMN_ID
        )
        diagnostic_data['predictions'] = self.estimator.predict(
            validation_data[[COLUMN_ERA] + feature_cols].to_numpy()
        )
        gc.collect()
        submit_diagnostic_predictions(diagnostic_data, self.numerai_model_id)

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


def main_arguments() -> Namespace:
    parser = argparse.ArgumentParser(description='Shrubbery')
    parser.add_argument(
        '--retrain', action='store_true', help='Use this flag to retrain'
    )
    return parser.parse_args()


@hydra.main(version_base=None, config_path='.', config_name='main')
def main(config: DictConfig) -> None:
    config_content = OmegaConf.to_yaml(config).encode('utf-8')
    runner: NumeraiRunner = hydra.utils.instantiate(config, _convert_='all')
    runner.run(config_content, 'run_config.yaml')


if __name__ == '__main__':
    main()
