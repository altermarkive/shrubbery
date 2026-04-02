from typing import Any, Callable

import numpy as np
import pandas as pd
import wandb

from shrubbery.constants import COLUMN_INDEX_TARGET


METRIC_PREDICTION_ID = 'Prediction ID'
METRIC_PREDICTION_VALUE = 'Metric'


# See also:
# - https://stackoverflow.com/questions/32401493/how-to-create-customize-your-own-scorer-function-in-scikit-learn  # noqa: E501
# - https://scikit-learn.org/stable/modules/model_evaluation.html
# - https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b/sklearn/metrics/_scorer.py#L604  # noqa: E501
def numerai_scorer(
    metric: Callable,
    greater_is_better: bool,
    name: str | None,
) -> Callable:
    ascending = 1.0 if greater_is_better else -1.0

    def scorer(estimator: Any, x: np.ndarray, y: np.ndarray) -> float:
        y_true = y
        if y.ndim > 1 and 1 not in y.shape:
            y_true = y_true[:, [COLUMN_INDEX_TARGET]]
        y_true = y_true.ravel()
        y_pred = estimator.predict(x)
        return ascending * metric(x, y_true, y_pred)

    if name is not None:
        scorer.__name__ = name  # type: ignore[attr-defined]
    return scorer


TABLE_EVALUATION = 'Evaluation Table'


def validation_metrics(
    x: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_function: Callable,
    validation_stats: list[dict[str, float]],
    prediction_id: str,
) -> None:
    evaluation: dict[str, Any] = {METRIC_PREDICTION_ID: prediction_id}
    result = metric_function(x, y_true, y_pred)
    evaluation[METRIC_PREDICTION_VALUE] = result
    validation_stats.append(evaluation)
    evaluation_table = pd.DataFrame.from_records(validation_stats)
    wandb.log({TABLE_EVALUATION: wandb.Table(data=evaluation_table)})
