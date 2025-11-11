#!/usr/bin/env python3

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Union

import pandas as pd

# from sklearn.metrics import mean_squared_error
import wandb
from numpy.typing import NDArray

from shrubbery.constants import COLUMN_INDEX_TARGET
from shrubbery.meta_estimator import NumeraiMetaEstimator
from shrubbery.metrics import (
    max_feature_exposure,
    per_era_max_apy,
    per_era_max_drawdown,
    per_era_sharpe,
)


@dataclass
class MetricConfig:
    metric_name: str
    greater_is_better: bool
    metric_function: Callable


METRIC_PREDICTION_ID = 'Prediction ID'


METRICS: List[MetricConfig] = [
    MetricConfig(
        'Sharpe',
        True,
        per_era_sharpe,
    ),
    MetricConfig(
        'Max Drawdown',
        True,
        per_era_max_drawdown,
    ),
    MetricConfig(
        'APY',
        True,
        per_era_max_apy,
    ),
    # MetricConfig(  # TODO: Commented out due to OOM - check if it happens after reboot  # noqa: E501
    #     'MSE',
    #     False,
    #     lambda _, y_true, y_pred: mean_squared_error(y_true, y_pred)
    # ),
    # Max Feature Exposure causes: RuntimeWarning: invalid value encountered in divide
    MetricConfig(
        'Max Feature Exposure',
        False,
        max_feature_exposure,
    ),
    # MetricConfig(
    #     ???,  # TODO: Create a function/constant for this
    #     ???,  # TODO: Create a function/constant for this
    #     lambda y_true, y_pred: partial(
    #         numerai_metrics, numerai_model_id=numerai_model_id
    #     ),
    # ),
]


# See also:
# - https://stackoverflow.com/questions/32401493/how-to-create-customize-your-own-scorer-function-in-scikit-learn  # noqa: E501
# - https://scikit-learn.org/stable/modules/model_evaluation.html
# - https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b/sklearn/metrics/_scorer.py#L604  # noqa: E501
def numerai_scorer(
    estimator: NumeraiMetaEstimator,
    x: NDArray,
    y: NDArray,
    metric: Callable,
    **kwargs: Dict[str, Any],
) -> float:
    y_true = y
    if y.ndim > 1 and 1 not in y.shape:
        y_true = y_true[:, [COLUMN_INDEX_TARGET]]
    y_true = y_true.ravel()
    y_pred = estimator.predict(x)
    return metric(x, y_true, y_pred)


def metric_to_ascending(metric: str) -> bool:
    for metric_config in METRICS:
        if metric == metric_config.metric_name:
            return not metric_config.greater_is_better
    raise NotImplementedError(f'Metric {metric} not found')


def _extract_metric_if_composite(
    metric: str, result: Union[float, Dict[str, float]]
) -> float:
    if isinstance(result, dict):
        return result[metric]
    return result


def metric_to_simple_scorer(metric: str) -> Callable:
    for metric_config in METRICS:
        if metric == metric_config.metric_name:
            ascending = 1.0 if metric_config.greater_is_better else -1.0
            scorer = partial(
                numerai_scorer,
                metric=lambda x, y_true, y_pred: ascending
                * _extract_metric_if_composite(
                    metric,
                    metric_config.metric_function(x, y_true, y_pred),
                ),
            )
            scorer.__name__ = metric  # type: ignore[attr-defined]
            return scorer
    raise NotImplementedError(f'Metric {metric} not found')


TABLE_EVALUATION = 'Evaluation Table'


def validation_metrics(
    x: NDArray,
    y_true: NDArray,
    y_pred: NDArray,
    validation_stats: List[Dict[str, float]],
    prediction_id: str,
) -> None:
    evaluation: Dict[str, Any] = {METRIC_PREDICTION_ID: prediction_id}
    for metric_config in METRICS:
        result = metric_config.metric_function(x, y_true, y_pred)
        evaluation[metric_config.metric_name] = result
    validation_stats.append(evaluation)
    evaluation_table = pd.DataFrame.from_records(validation_stats)
    wandb.log({TABLE_EVALUATION: wandb.Table(data=evaluation_table)})
