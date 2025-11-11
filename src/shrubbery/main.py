import gc
from typing import Any, Dict, List, Optional

import hydra
import hydra.utils
import numpy as np
import pandas as pd
import wandb
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin
from sklearn.model_selection import GridSearchCV

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
from shrubbery.evaluation import metric_to_simple_scorer
from shrubbery.meta_estimator import NumeraiMetaEstimator
from shrubbery.napi import napi
from shrubbery.observability import logger
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
        numerai_model_id: str,
        model_name: str,
        drop_era_column: bool,
        downsample_cross_validation: int,
        downsample_full_train: int,
        targets: list[int | str],
        cv: int,
        cv_param_grid: Dict,
        cv_metric: str,
        embargo: int,
        neutralization_feature_indices: Optional[List[int]],
        neutralization_proportion: float,
        neutralization_normalize: bool,
        tb: int,
    ) -> None:
        self.estimator = NumeraiMetaEstimator(
            estimator=estimator,
            drop_era_column=drop_era_column,
            target=COLUMN_INDEX_TARGET,
            neutralization_feature_indices=neutralization_feature_indices,
            neutralization_proportion=neutralization_proportion,
            neutralization_normalize=neutralization_normalize,
        )
        self.numerai_model_id = numerai_model_id
        self.model_name = model_name
        self.drop_era_column = drop_era_column
        self.downsample_cross_validation = downsample_cross_validation
        self.downsample_full_train = downsample_full_train
        self.targets = targets
        self.cv = cv
        self.cv_param_grid = cv_param_grid
        self.cv_metric = cv_metric
        self.embargo = embargo
        self.neutralization_feature_indices = neutralization_feature_indices
        self.neutralization_proportion = neutralization_proportion
        self.neutralization_normalize = neutralization_normalize
        self.tb = tb

    def fit(
        self, x: NDArray, y: NDArray, **kwargs: Dict[str, Any]
    ) -> NumeraiMetaEstimator:
        logger.info(f'Shape of training data - x:{x.shape} y:{y.shape}')
        targets = [
            target if isinstance(target, int) else lookup_target_index(target)
            for target in self.targets
        ]
        y = y[:, targets]

        # Do cross val to get out of sample training preds
        # Get out of sample training preds via embargoed
        # time series cross validation
        logger.info('Entering time series cross validation loop')
        grid_search_cv = GridSearchCV(
            self.estimator,
            param_grid=self.cv_param_grid | {'target': range(y.shape[1])},
            cv=NumeraiTimeSeriesSplitter(
                cv=self.cv,
                embargo=self.embargo,
            ),
            # In case multiple scores are of interest, see: https://stackoverflow.com/questions/35876508/evaluate-multiple-scores-on-sklearn-cross-val-score & https://scikit-learn.org/stable/modules/grid_search.html#composite-grid-search  # noqa: E501
            scoring=metric_to_simple_scorer(self.cv_metric),
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
        best_model_parameters: Dict = {
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

    def predict(self, x: NDArray) -> NDArray:
        return self.estimator.predict(x)


class NumeraiRunner:
    def __init__(
        self,
        feature_set_name: str,
        retrain: bool,
        estimator: Any,
        numerai_model_id: str,
        version: str,
        extra_training_targets: int,
        notes: str,
    ) -> None:
        self.feature_set_name = feature_set_name
        self.retrain = retrain
        self.estimator = estimator
        self.numerai_model_id = numerai_model_id
        self.version = version
        self.extra_training_targets = extra_training_targets
        self.notes = notes

    def run(self) -> None:
        if wandb.run is not None:
            tags = list(wandb.run.tags) if wandb.run.tags else []
            wandb.run.tags = tuple(
                tags + [f'numerai_model_id:{self.numerai_model_id}']
            )
            wandb.run.summary.update(
                {'numerai_model_id': self.numerai_model_id}
            )
            wandb.run.notes = self.notes
        download_numerai_files()
        feature_cols = get_feature_set(self.feature_set_name)
        targets = get_training_targets()
        read_columns = [COLUMN_ERA] + feature_cols + targets

        logger.info('Reading training data')
        training_data, training_eras = read_parquet_and_unpack(
            'train.parquet', read_columns, feature_cols
        )
        logger.info('Reading validation data')
        validation_data, validation_eras = read_parquet_and_unpack(
            'validation.parquet', read_columns, feature_cols
        )
        logger.info('Reading tournament data')
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
        model, version = (
            (None, 'latest')
            if self.retrain
            else load_model(model_name, self.version)
        )
        if model is None:
            # Now do a full train
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

        # Garbage collection gets rid of unused data and frees up memory
        gc.collect()

        prediction_data = pd.DataFrame(live_data.index).set_index(COLUMN_ID)
        prediction_data['predictions'] = self.estimator.predict(
            live_data[[COLUMN_ERA] + feature_cols].to_numpy()
        )
        gc.collect()

        submit_tournament_predictions(prediction_data, self.numerai_model_id)


def _save_config_file_to_wandb(config: DictConfig) -> None:
    directory = get_workspace_path()
    config_path = directory / 'run_config.yaml'
    with open(config_path, 'wb') as handle:
        handle.write(OmegaConf.to_yaml(config).encode('utf-8'))
    wandb.save(config_path, base_path=directory)


@hydra.main(version_base=None, config_path='.', config_name='main')
def main(config: DictConfig) -> None:
    # W&B Tags
    tags = []
    try:
        round = napi.get_current_round()
        tags.append(str(round))
    except ValueError:
        pass
    runner: NumeraiRunner = hydra.utils.instantiate(config, _convert_='all')
    update_tournament_submissions(runner.numerai_model_id)
    wandb.init(tags=tags)
    _save_config_file_to_wandb(config)
    runner.run()
    wandb.finish()


if __name__ == '__main__':
    # # To prevent crash/freeze with n_jobs > 1
    # # (hinted at in TPOT's documentation)
    # import multiprocessing
    # multiprocessing.set_start_method('forkserver')
    main()
