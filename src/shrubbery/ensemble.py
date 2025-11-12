import gc
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import numpy as np
import pandas as pd
import wandb
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin

from shrubbery.constants import COLUMN_INDEX_TARGET
from shrubbery.evaluation import METRIC_PREDICTION_VALUE, validation_metrics
from shrubbery.mixer import mix_combinatorial, mix_predictions
from shrubbery.observability import logger
from shrubbery.utilities import PrintableModelMixin, model_to_string


class EnsembleType(str, Enum):
    PRODUCT_AND_ROOT = 'product_and_root'
    SUM_AND_RANK = 'sum_and_rank'


def get_ensemble(
    ensemble_type: EnsembleType,
) -> Callable[[np.ndarray], np.ndarray]:
    if ensemble_type == EnsembleType.PRODUCT_AND_ROOT:
        return ensemble_product_and_root
    elif ensemble_type == EnsembleType.SUM_AND_RANK:
        return ensemble_sum_and_rank
    else:
        raise ValueError(f'Unknown ensemble type: {ensemble_type}')


# Inspired by: https://github.com/jimfleming/numerai/blob/master/ensemble.py#L22  # noqa: E501
def ensemble_product_and_root(y_preds: np.ndarray) -> np.ndarray:
    return np.power(np.prod(y_preds, axis=1), 1.0 / y_preds.shape[1])


def ensemble_sum_and_rank(y_preds: np.ndarray) -> np.ndarray:
    return pd.DataFrame(y_preds).sum(axis=1).rank(pct=True).to_numpy()


METRIC = 'Metric'


@dataclass
class EstimatorConfig:
    name: str
    estimator: Any


class Ensembler(
    BaseEstimator, MetaEstimatorMixin, RegressorMixin, PrintableModelMixin
):
    def __init__(
        self,
        estimators: list[EstimatorConfig],
        numerai_model_id: str,
        ensemble_metric_function: Callable,
        ensemble_metric_greater_is_better: bool,
        ensemble_type: EnsembleType,
        mix_combinatorial_cap: int | None,
    ) -> None:
        self.estimators = estimators
        self.numerai_model_id = numerai_model_id
        self.ensemble_metric_function = ensemble_metric_function
        self.ensemble_metric_greater_is_better = (
            ensemble_metric_greater_is_better
        )
        self.ensemble_type = ensemble_type
        self.mix_combinatorial_cap = mix_combinatorial_cap
        self.estimator_names_best_ = [config.name for config in estimators]

    def fit(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict[str, Any]
    ) -> 'Ensembler':
        x_training = x
        y_training = y
        for config in self.estimators:
            # Now do a full train
            logger.info(f'Training ensemble model: {config.name}')
            config.estimator = config.estimator.fit(x_training, y_training)
            # Garbage collection gets rid of unused data and frees up memory
            gc.collect()
        # Keep track of prediction columns and stats
        predictions: dict[str, np.ndarray] = {}
        validation_stats: list[dict[str, float]] = []
        for config in self.estimators:
            logger.info(f'Predicting ensemble model: {config.name}')
            logger.info(
                f'Ensemble model config: {model_to_string(config.estimator)}'
            )
            y_predictions = config.estimator.predict(x_training)
            predictions[config.name] = y_predictions
            validation_metrics(
                x_training,
                y_training[:, COLUMN_INDEX_TARGET].ravel(),
                y_predictions,
                self.ensemble_metric_function,
                validation_stats,
                config.name,
            )
            gc.collect()
        logger.info('Creating ensemble for validation')
        ensemble_metric_function = self.ensemble_metric_function
        ensemble_metric_ascending = not self.ensemble_metric_greater_is_better
        best = mix_combinatorial(
            x_training,
            y_training,
            predictions,
            ensemble_metric_function,
            validation_stats,
            get_ensemble(self.ensemble_type),
            sort_by=METRIC_PREDICTION_VALUE,
            sort_ascending=ensemble_metric_ascending,
            cap=self.mix_combinatorial_cap,
        )
        gc.collect()
        if best:
            logger.info(f'Ensemble with highest score: {best}')
            self.estimator_names_best_ = best
            if wandb.run is not None:
                wandb.run.summary.update({'best_model': ' '.join(best)})
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions: dict[str, np.ndarray] = {}
        for config in self.estimators:
            if config.name in self.estimator_names_best_:
                logger.info(f'Predicting ensemble model: {config.name}')
                logger.info(
                    f'Ensemble model config: {model_to_string(config.estimator)}'
                )
                predictions[config.name] = config.estimator.predict(
                    x.astype(np.float32)
                )
        logger.info('Creating ensemble for tournament')
        logger.info(f'Ensemble: {self.estimator_names_best_}')
        ensemble = get_ensemble(self.ensemble_type)
        return mix_predictions(
            predictions, self.estimator_names_best_, ensemble
        )
