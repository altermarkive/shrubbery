#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin

from shrubbery.constants import (
    COLUMN_DATA_TYPE_TRAINING,
    COLUMN_DATA_TYPE_VALIDATION,
    COLUMN_INDEX_DATA_TYPE,
    COLUMN_INDEX_EXAMPLE_PREDICTIONS,
    COLUMN_INDEX_TARGET,
)
from shrubbery.evaluation import metric_to_ascending, validation_metrics
from shrubbery.mixer import mix_combinatorial, mix_predictions
from shrubbery.observability import logger
from shrubbery.utilities import PrintableModelMixin

import wandb  # isort: skip


class EnsembleType(str, Enum):
    PRODUCT_AND_ROOT = 'product_and_root'
    SUM_AND_RANK = 'sum_and_rank'


def get_ensemble(
    ensemble_type: EnsembleType,
) -> Callable[[NDArray], NDArray]:
    if ensemble_type == EnsembleType.PRODUCT_AND_ROOT:
        return ensemble_product_and_root
    elif ensemble_type == EnsembleType.SUM_AND_RANK:
        return ensemble_sum_and_rank
    else:
        raise ValueError(f'Unknown ensemble type: {ensemble_type}')


# Inspired by: https://github.com/jimfleming/numerai/blob/master/ensemble.py#L22  # noqa: E501
def ensemble_product_and_root(y_preds: NDArray) -> NDArray:
    return np.power(np.prod(y_preds, axis=1), 1.0 / y_preds.shape[1])


def ensemble_sum_and_rank(y_preds: NDArray) -> NDArray:
    return pd.DataFrame(y_preds).sum(axis=1).rank(pct=True).to_numpy()


@dataclass
class EstimatorConfig:
    name: str
    estimator: Any


class Ensembler(
    BaseEstimator, MetaEstimatorMixin, RegressorMixin, PrintableModelMixin
):
    def __init__(
        self,
        estimators: List[EstimatorConfig],
        numerai_model_id: str,
        training_metric: str,
        ensemble_metric: str,
        ensemble_type: EnsembleType,
        mix_combinatorial_cap: Optional[int],
        neutralization_feature_indices: List[int],
        neutralization_proportion: float,
        neutralization_normalize: bool,
        tb: int,
    ) -> None:
        self.estimators = estimators
        self.numerai_model_id = numerai_model_id
        self.training_metric = training_metric
        self.ensemble_metric = ensemble_metric
        self.ensemble_type = ensemble_type
        self.mix_combinatorial_cap = mix_combinatorial_cap
        self.neutralization_feature_indices = neutralization_feature_indices
        self.neutralization_proportion = neutralization_proportion
        self.neutralization_normalize = neutralization_normalize
        self.tb = tb
        self.estimator_names_best_ = [config.name for config in estimators]

    def fit(
        self, x: NDArray, y: NDArray, **kwargs: Dict[str, Any]
    ) -> 'Ensembler':
        training_data_selection = (
            x[:, COLUMN_INDEX_DATA_TYPE] == COLUMN_DATA_TYPE_TRAINING
        )
        validation_data_selection = (
            x[:, COLUMN_INDEX_DATA_TYPE] == COLUMN_DATA_TYPE_VALIDATION
        )
        x_training = x[
            training_data_selection, :COLUMN_INDEX_DATA_TYPE
        ].astype(np.float32)
        y_training = y[
            training_data_selection, :COLUMN_INDEX_EXAMPLE_PREDICTIONS
        ].astype(np.float32)
        for config in self.estimators:
            # Now do a full train
            logger.info(f'Training model: {config.name}')
            config.estimator = config.estimator.fit(x_training, y_training)
            # Garbage collection gets rid of unused data and frees up memory
            gc.collect()
        # Keep track of prediction columns and stats
        predictions: Dict[str, NDArray] = {}
        validation_stats: List[Dict[str, float]] = []
        x_validation = x[
            validation_data_selection, :COLUMN_INDEX_DATA_TYPE
        ].astype(np.float32)
        y_true = y[validation_data_selection, [COLUMN_INDEX_TARGET]].astype(
            np.float32
        )
        examples = y[
            validation_data_selection, COLUMN_INDEX_EXAMPLE_PREDICTIONS:
        ].astype(np.float32)
        for config in self.estimators:
            logger.info(f'Predicting model: {config.name}')
            logger.info(f'Model config: {config.estimator}')
            y_predictions = config.estimator.predict(x_validation)
            predictions[config.name] = y_predictions
            validation_metrics(
                x_validation,
                y_true,
                y_predictions,
                examples,
                validation_stats,
                config.name,
                numerai_model_id=self.numerai_model_id,
                neutralization_feature_indices=(
                    self.neutralization_feature_indices
                ),
                neutralization_proportion=self.neutralization_proportion,
                neutralization_normalize=self.neutralization_normalize,
                tb=self.tb,
            )
            gc.collect()
        logger.info('Creating ensemble for validation')
        ensemble_metric = self.ensemble_metric
        ensemble_metric_ascending = metric_to_ascending(
            ensemble_metric,
            self.numerai_model_id,
            self.neutralization_feature_indices,
            self.neutralization_proportion,
            self.neutralization_normalize,
            self.tb,
        )
        best = mix_combinatorial(
            x_validation,
            y_true,
            examples,
            predictions,
            validation_stats,
            self.numerai_model_id,
            get_ensemble(self.ensemble_type),
            sort_by=ensemble_metric,
            sort_ascending=ensemble_metric_ascending,
            cap=self.mix_combinatorial_cap,
            neutralization_feature_indices=self.neutralization_feature_indices,
            neutralization_proportion=self.neutralization_proportion,
            neutralization_normalize=self.neutralization_normalize,
            tb=self.tb,
        )
        gc.collect()
        if best:
            logger.info(f'Ensemble with highest score: {best}')
            self.estimator_names_best_ = best
            if wandb.run is not None:
                wandb.run.summary.update({'best_model': ' '.join(best)})
        return self

    def predict(self, x: NDArray) -> NDArray:
        predictions: Dict[str, NDArray] = {}
        for config in self.estimators:
            if config.name in self.estimator_names_best_:
                logger.info(f'Predicting model: {config.name}')
                logger.info(f'Model config: {config.estimator}')
                predictions[config.name] = config.estimator.predict(
                    x[:, :COLUMN_INDEX_DATA_TYPE].astype(np.float32)
                )
        logger.info('Creating ensemble for tournament')
        logger.info(f'Ensemble: {self.estimator_names_best_}')
        ensemble = get_ensemble(self.ensemble_type)
        return mix_predictions(
            predictions, self.estimator_names_best_, ensemble
        )
