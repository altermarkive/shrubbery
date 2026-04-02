from typing import Callable

import numpy as np
import pandas as pd

from shrubbery.evaluation import METRIC_PREDICTION_ID, validation_metrics
from shrubbery.observability import logger


def _encode(decoded: set[str]) -> str:
    return '_'.join(sorted(list(decoded)))


def _decode(encoded: str) -> set[str]:
    return set(encoded.split('_'))


def _sort_reports(lut: dict, ascending: bool) -> list:
    return sorted(lut.items(), key=lambda item: item[1], reverse=not ascending)


def next_mix(lut: dict, ascending: bool) -> str | None:
    reports = _sort_reports(lut, ascending)
    length = len(reports)
    if length < 2:
        return None
    for i in range(length - 1):
        a = _decode(reports[i][0])
        for j in range(i + 1, length):
            b = _decode(reports[j][0])
            ab = _encode(a.union(b))
            if ab not in lut:
                return ab
    return None


def top_mix(lut: dict, ascending: bool) -> str | None:
    reports = _sort_reports(lut, ascending)
    return reports[0][0] if reports else None


def mix_predictions(
    predictions: dict[str, np.ndarray],
    pred_cols: list[str],
    ensemble: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    return ensemble(
        np.concatenate(
            [predictions[pred_col].reshape(-1, 1) for pred_col in pred_cols],
            axis=1,
        )
    )


def mix_all(
    x: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    ensemble_metric_function: Callable,
    validation_stats: list[dict[str, float]],
    pred_cols: list[str],
    ensemble: Callable[[np.ndarray], np.ndarray],
) -> None:
    predictions_name = _encode(set(pred_cols))
    y_prediction = mix_predictions(predictions, pred_cols, ensemble)
    predictions[predictions_name] = y_prediction
    validation_metrics(
        x,
        y_true,
        y_prediction,
        ensemble_metric_function,
        validation_stats,
        predictions_name,
    )


def mix_combinatorial(
    x: np.ndarray,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    ensemble_metric_function: Callable,
    validation_stats: list[dict[str, float]],
    ensemble: Callable[[np.ndarray], np.ndarray],
    sort_by: str,
    sort_ascending: bool,
    cap: int | None,
) -> list[str] | None:
    lut = {
        item[METRIC_PREDICTION_ID]: item[sort_by]
        for item in validation_stats
        if item[METRIC_PREDICTION_ID] in predictions.keys()
    }
    if cap is None:
        cap = 2 ** len(predictions.keys())
    for _ in range(cap):
        mix = next_mix(lut, sort_ascending)
        if mix is None:
            break
        logger.info(f'Next up: {mix}')
        mix_all(
            x,
            y_true,
            predictions,
            ensemble_metric_function,
            validation_stats,
            list(_decode(mix)),
            ensemble,
        )
        lut = {
            item[METRIC_PREDICTION_ID]: item[sort_by]
            for item in validation_stats
        }
        ranking = (
            pd.DataFrame(
                lut.items(), columns=pd.Series(['Prediction', sort_by])
            )
            .sort_values(by=sort_by, ascending=sort_ascending)
        )
        for _, row in ranking.iterrows():
            logger.info(f'Ranking: {row[sort_by]} {row["Prediction"]}')
    top = top_mix(lut, sort_ascending)
    if top:
        return list(_decode(top))
    return None
