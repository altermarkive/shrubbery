#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Union

import pandas as pd
from numpy.typing import NDArray

# from sklearn.metrics import mean_squared_error
import wandb
from shrubbery.constants import (
    COLUMN_EXAMPLE_PREDICTIONS,
    COLUMN_INDEX_TARGET,
    COLUMN_Y_PRED,
    COLUMN_Y_TRUE,
)
from shrubbery.meta_estimator import NumeraiMetaEstimator
from shrubbery.metrics import (
    METRIC_APY,
    METRIC_CORR_WITH_EXAMPLE_PREDICTIONS,
    METRIC_EXPOSURE_DISSIMILARITY_MEAN,
    METRIC_FEATURE_NEUTRAL_MEAN,
    METRIC_MAX_DRAWDOWN,
    METRIC_MAX_FEATURE_EXPOSURE,
    METRIC_MMC_CORR_SHARPE,
    METRIC_MMC_MEAN,
    METRIC_SHARPE_MEAN,
    METRIC_SHARPE_SD,
    METRIC_SHARPE_VALUE,
    METRIC_TB_FEATURE_NEUTRAL_MEAN,
    METRIC_TB_MEAN,
    METRIC_TB_SD,
    METRIC_TB_SHARPE,
    corr_with_example_predictions,
    exposure_dissimilarity_mean,
    feature_neutral_mean,
    max_feature_exposure,
    mmc_metrics,
    per_era_max_apy,
    per_era_max_drawdown,
    per_era_sharpe,
    tb_fast_score_by_date,
    tb_feature_neutral_mean,
)


@dataclass
class MetricConfig:
    metric_names: List[str]
    greater_is_better: List[bool]
    metric_function: Callable
    metric_function_arguments: List[str]


METRIC_PREDICTION_ID = 'Prediction ID'

ARGUMENT_X = 'x'
ARGUMENT_Y_TRUE = COLUMN_Y_TRUE
ARGUMENT_Y_PRED = COLUMN_Y_PRED
ARGUMENT_EXAMPLES = COLUMN_EXAMPLE_PREDICTIONS

ARGUMENTS_DEFAULT = [ARGUMENT_Y_TRUE, ARGUMENT_Y_PRED]
ARGUMENTS_WITH_X = [ARGUMENT_X, ARGUMENT_Y_TRUE, ARGUMENT_Y_PRED]
ARGUMENTS_WITH_X_AND_EXAMPLES = [
    ARGUMENT_X,
    ARGUMENT_Y_TRUE,
    ARGUMENT_Y_PRED,
    ARGUMENT_EXAMPLES,
]


def _metrics(
    numerai_model_id: str,
    neutralization_feature_indices: List[int],
    neutralization_proportion: float,
    neutralization_normalize: bool,
    tb: int,
) -> List[MetricConfig]:
    metrics: List[MetricConfig] = [
        MetricConfig(
            [METRIC_SHARPE_MEAN, METRIC_SHARPE_SD, METRIC_SHARPE_VALUE],
            [True, False, True],
            per_era_sharpe,
            ARGUMENTS_WITH_X,
        ),
        MetricConfig(
            [METRIC_MAX_DRAWDOWN],
            [True],
            per_era_max_drawdown,
            ARGUMENTS_WITH_X,
        ),
        MetricConfig(
            [METRIC_APY],
            [True],
            per_era_max_apy,
            ARGUMENTS_WITH_X,
        ),
        # MetricConfig(  # TODO: Commented out due to OOM - check if it happens after reboot  # noqa: E501
        #     [METRIC_MSE],
        #     [False],
        #     mean_squared_error,
        #     ARGUMENTS_DEFAULT,
        # ),
        MetricConfig(
            [METRIC_MAX_FEATURE_EXPOSURE],
            [False],
            max_feature_exposure,
            ARGUMENTS_WITH_X,
        ),
        MetricConfig(
            [METRIC_FEATURE_NEUTRAL_MEAN],
            [True],
            partial(
                feature_neutral_mean,
                neutralization_feature_indices=neutralization_feature_indices,
                neutralization_proportion=neutralization_proportion,
                neutralization_normalize=neutralization_normalize,
            ),
            ARGUMENTS_WITH_X,
        ),
        MetricConfig(
            [METRIC_TB_FEATURE_NEUTRAL_MEAN],
            [True],
            partial(
                tb_feature_neutral_mean,
                neutralization_feature_indices=neutralization_feature_indices,
                neutralization_proportion=neutralization_proportion,
                neutralization_normalize=neutralization_normalize,
                tb=tb,
            ),
            ARGUMENTS_WITH_X,
        ),
        MetricConfig(
            [METRIC_TB_MEAN, METRIC_TB_SD, METRIC_TB_SHARPE],
            [True, False, True],
            partial(tb_fast_score_by_date, tb=tb),
            ARGUMENTS_WITH_X,
        ),
        MetricConfig(
            [METRIC_MMC_MEAN, METRIC_MMC_CORR_SHARPE],
            [True, True],
            partial(
                mmc_metrics,
                neutralization_proportion=neutralization_proportion,
            ),
            ARGUMENTS_WITH_X_AND_EXAMPLES,
        ),
        MetricConfig(
            [METRIC_CORR_WITH_EXAMPLE_PREDICTIONS],
            [True],
            corr_with_example_predictions,
            ARGUMENTS_WITH_X_AND_EXAMPLES,
        ),
        MetricConfig(
            [METRIC_EXPOSURE_DISSIMILARITY_MEAN],
            [False],
            exposure_dissimilarity_mean,
            ARGUMENTS_WITH_X_AND_EXAMPLES,
        ),
        # MetricConfig(
        #     [???],  # TODO: Create a function/constant for this
        #     [???],  # TODO: Create a function/constant for this
        #     lambda y_true, y_pred: partial(
        #         numerai_metrics, numerai_model_id=numerai_model_id
        #     ),
        #     ARGUMENTS_DEFAULT
        # ),
    ]
    return metrics


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


def metric_to_ascending(
    metric: str,
    numerai_model_id: str,
    neutralization_feature_indices: List[int],
    neutralization_proportion: float,
    neutralization_normalize: bool,
    tb: int,
) -> bool:
    for metric_config in _metrics(
        numerai_model_id=numerai_model_id,
        neutralization_feature_indices=neutralization_feature_indices,
        neutralization_proportion=neutralization_proportion,
        neutralization_normalize=neutralization_normalize,
        tb=tb,
    ):
        if metric in metric_config.metric_names:
            return not metric_config.greater_is_better[
                metric_config.metric_names.index(metric)
            ]
    raise NotImplementedError(f'Metric {metric} not found')


def _extract_metric_if_composite(
    metric: str, result: Union[float, Dict[str, float]]
) -> float:
    if isinstance(result, dict):
        return result[metric]
    return result


def metric_to_simple_scorer(
    metric: str,
    numerai_model_id: str,
    neutralization_feature_indices: List[int],
    neutralization_proportion: float,
    neutralization_normalize: bool,
    tb: int,
) -> Callable:
    metrics = _metrics(
        numerai_model_id=numerai_model_id,
        neutralization_feature_indices=neutralization_feature_indices,
        neutralization_proportion=neutralization_proportion,
        neutralization_normalize=neutralization_normalize,
        tb=tb,
    )
    for metric_config in metrics:
        if metric in metric_config.metric_names:
            ascending = 1.0 if metric_config.greater_is_better else -1.0
            if metric_config.metric_function_arguments == ARGUMENTS_DEFAULT:
                scorer = partial(
                    numerai_scorer,
                    metric=lambda x, y_true, y_pred: ascending
                    * metric_config.metric_function(y_true, y_pred),
                )
                scorer.__name__ = metric  # type: ignore[attr-defined]
                return scorer
            elif metric_config.metric_function_arguments == ARGUMENTS_WITH_X:
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
            else:
                NotImplementedError(
                    f'Metric {metric} cannot be used as a simple scorer'
                )
    raise NotImplementedError(f'Metric {metric} not found')


TABLE_EVALUATION = 'Evaluation Table'


def validation_metrics(
    x: NDArray,
    y_true: NDArray,
    y_pred: NDArray,
    examples: NDArray,
    validation_stats: List[Dict[str, float]],
    prediction_id: str,
    numerai_model_id: str,
    neutralization_feature_indices: List[int],
    neutralization_proportion: float,
    neutralization_normalize: bool,
    tb: int,
) -> None:
    evaluation: Dict[str, Any] = {METRIC_PREDICTION_ID: prediction_id}
    arguments = {
        ARGUMENT_X: x,
        ARGUMENT_Y_TRUE: y_true,
        ARGUMENT_Y_PRED: y_pred,
        ARGUMENT_EXAMPLES: examples,
    }
    for metric_config in _metrics(
        numerai_model_id=numerai_model_id,
        neutralization_feature_indices=neutralization_feature_indices,
        neutralization_proportion=neutralization_proportion,
        neutralization_normalize=neutralization_normalize,
        tb=tb,
    ):
        result = metric_config.metric_function(
            **{
                key: arguments[key]
                for key in metric_config.metric_function_arguments
            }
        )
        if isinstance(result, float):
            evaluation[metric_config.metric_names[0]] = result
        elif isinstance(result, dict):
            for metric_name in result:
                evaluation[metric_name] = result[metric_name]
    validation_stats.append(evaluation)
    evaluation_table = pd.DataFrame.from_records(validation_stats)
    evaluation_table[METRIC_PREDICTION_ID] = evaluation_table[
        METRIC_PREDICTION_ID
    ].apply(lambda name: f'{numerai_model_id}_{name}')
    wandb.log({TABLE_EVALUATION: wandb.Table(data=evaluation_table)})
