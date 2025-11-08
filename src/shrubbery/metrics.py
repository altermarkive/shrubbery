#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray

from shrubbery.constants import (
    COLUMN_ERA,
    COLUMN_ID,
    COLUMN_INDEX_ERA,
    COLUMN_Y_PRED,
    COLUMN_Y_TRUE,
)
from shrubbery.data.augmentation import FILE_VALIDATION_IDS
from shrubbery.data.ingest import locate_numerai_file
from shrubbery.napi import napi
from shrubbery.neutralization import neutralize
from shrubbery.observability import logger
from shrubbery.utilities import save_prediction


def _compose_metric_name(
    abstract_metric_name: str, sub_metric_name: str
) -> str:
    return f'{abstract_metric_name} {sub_metric_name}'.strip()


def _unif(df: pd.DataFrame) -> pd.Series:
    x = (df.rank(method='first') - 0.5) / len(df)
    return pd.Series(x, index=df.index)


def _calculate_validation_correlations(
    x: NDArray, y_true: NDArray, y_pred: NDArray
) -> pd.DataFrame:
    validation_data = pd.DataFrame(
        np.concatenate(
            [
                x[:, COLUMN_INDEX_ERA].reshape(-1, 1),
                y_true.reshape(-1, 1),
                y_pred.reshape(-1, 1),
            ],
            axis=1,
        )
    ).set_axis([COLUMN_ERA, COLUMN_Y_TRUE, COLUMN_Y_PRED], axis=1)
    validation_correlations = validation_data.groupby(
        COLUMN_ERA, group_keys=False
    ).apply(
        lambda group: _unif(group[COLUMN_Y_PRED]).corr(group[COLUMN_Y_TRUE]),
        include_groups=False,
    )  # type: ignore[index]
    return validation_correlations


def _get_validation_data_grouped(
    x: NDArray, y_true: NDArray, y_pred: NDArray
) -> Tuple[pd.DataFrame, Sequence[int]]:
    feature_indices = list(range(COLUMN_INDEX_ERA + 1, x.shape[1]))
    validation_data = pd.DataFrame(
        np.concatenate(
            [
                x,
                y_true.reshape(-1, 1),
                y_pred.reshape(-1, 1),
            ],
            axis=1,
        )
    )
    columns = validation_data.columns.to_list()
    columns[COLUMN_INDEX_ERA] = COLUMN_ERA
    columns[-2] = COLUMN_Y_TRUE
    columns[-1] = COLUMN_Y_PRED
    validation_data = validation_data.set_axis(columns, axis=1)
    return (
        validation_data.groupby(COLUMN_ERA, group_keys=False),
        feature_indices,
    )


ABSTRACT_METRIC_SHARPE = 'Sharpe'
METRIC_SHARPE_MEAN = _compose_metric_name(ABSTRACT_METRIC_SHARPE, 'Mean')
METRIC_SHARPE_SD = _compose_metric_name(ABSTRACT_METRIC_SHARPE, 'SD')
METRIC_SHARPE_VALUE = _compose_metric_name(ABSTRACT_METRIC_SHARPE, '')


# Numerai-specific sharpe ratio scorer
def per_era_sharpe(x: NDArray, y_true: NDArray, y_pred: NDArray) -> float:
    validation_correlations = _calculate_validation_correlations(
        x, y_true, y_pred
    )
    mean = validation_correlations.mean()
    std = validation_correlations.std(ddof=0)
    sharpe = mean / std
    return sharpe


METRIC_MAX_DRAWDOWN = 'Max Drawdown'


def per_era_max_drawdown(
    x: NDArray, y_true: NDArray, y_pred: NDArray
) -> float:
    validation_correlations = _calculate_validation_correlations(
        x, y_true, y_pred
    )
    rolling_max = (
        (validation_correlations + 1)
        .cumprod()
        .rolling(window=9000, min_periods=1)  # arbitrarily large
        .max()
    )
    daily_value = (validation_correlations + 1).cumprod()
    max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
    return max_drawdown


METRIC_APY = 'APY'


def per_era_max_apy(x: NDArray, y_true: NDArray, y_pred: NDArray) -> float:
    validation_correlations = _calculate_validation_correlations(
        x, y_true, y_pred
    )
    payout_scores = validation_correlations.clip(-0.25, 0.25)
    payout_daily_value = (payout_scores + 1).cumprod()
    apy = (
        ((payout_daily_value.dropna().iloc[-1]) ** (1 / len(payout_scores)))
        ** 49  # 52 weeks of compounding minus 3 for stake compounding lag
        - 1
    ) * 100
    return apy


METRIC_MAX_FEATURE_EXPOSURE = 'Max Feature Exposure'


def max_feature_exposure(
    x: NDArray, y_true: NDArray, y_pred: NDArray
) -> float:
    # Check the feature exposure of your validation predictions
    validation_data_grouped, feature_indices = _get_validation_data_grouped(
        x, y_true, y_pred
    )
    max_per_era = validation_data_grouped.apply(
        lambda group: group[feature_indices]
        .corrwith(group[COLUMN_Y_PRED])
        .abs()
        .max(),
        include_groups=False,
    )
    return max_per_era.mean()


METRIC_MSE = 'MSE'


def numerai_metrics(
    y_true: NDArray, y_pred: NDArray, numerai_model_id: str
) -> Dict[str, float]:
    prediction_data = pd.read_csv(
        locate_numerai_file(FILE_VALIDATION_IDS), index_col=COLUMN_ID
    )
    prediction_data['predictions'] = y_pred
    prediction_name = 'validation'
    prediction_path = save_prediction(prediction_data, prediction_name)
    # Upload validation prediction (Scores -> Models -> Run Diagnostics)
    while True:
        try:
            logger.info('Uploading prediction')
            diagnostics_id = napi.upload_diagnostics(
                file_path=str(prediction_path),
                model_id=numerai_model_id,
            )
            logger.info('Uploaded prediction')
            break
        except requests.exceptions.HTTPError as error:
            if (
                error.response is not None
                and error.response.status_code == 429
            ):
                logger.info('Backing off')
                time.sleep(30 * 60)
            else:
                logger.exception('Network failure')
                time.sleep(60)
        except Exception:
            logger.exception('Upload failure')
            time.sleep(10)
    # Fetch diagnostics
    while True:
        diagnostics = napi.diagnostics(
            model_id=numerai_model_id, diagnostics_id=diagnostics_id
        )[0]
        if diagnostics['status'] == 'done':
            break
        time.sleep(10)
    metrics = {}
    for key, value in diagnostics.items():
        if isinstance(value, float) or isinstance(value, int):
            metrics[f'Numerai {key}'] = float(value)
    return metrics
