from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shrubbery.evaluation import METRIC_PREDICTION_ID, validation_metrics
from shrubbery.observability import logger


def _encode(decoded: Set[str]) -> str:
    return '_'.join(sorted(list(decoded)))


def _decode(encoded: str) -> Set[str]:
    return set(encoded.split('_'))


def _sort_reports(lut: Dict, ascending: bool) -> List:
    return sorted(lut.items(), key=lambda item: item[1], reverse=not ascending)


def next_mix(lut: Dict, ascending: bool) -> Optional[str]:
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


def top_mix(lut: Dict, ascending: bool) -> Optional[str]:
    reports = _sort_reports(lut, ascending)
    return reports[0][0] if reports else None


def mix_predictions(
    predictions: Dict[str, NDArray],
    pred_cols: List[str],
    ensemble: Callable[[NDArray], NDArray],
) -> NDArray:
    return ensemble(
        np.concatenate(
            [predictions[pred_col].reshape(-1, 1) for pred_col in pred_cols],
            axis=1,
        )
    )


def mix_all(
    x: NDArray,
    y_true: NDArray,
    predictions: Dict[str, NDArray],
    validation_stats: List[Dict[str, float]],
    pred_cols: List[str],
    ensemble: Callable[[NDArray], NDArray],
) -> None:
    predictions_name = _encode(set(pred_cols))
    y_prediction = mix_predictions(predictions, pred_cols, ensemble)
    predictions[predictions_name] = y_prediction
    validation_metrics(
        x,
        y_true,
        y_prediction,
        validation_stats,
        predictions_name,
    )


def mix_combinatorial(
    x: NDArray,
    y_true: NDArray,
    predictions: Dict[str, NDArray],
    validation_stats: List[Dict[str, float]],
    ensemble: Callable[[NDArray], NDArray],
    sort_by: str,
    sort_ascending: bool,
    cap: Optional[int],
    neutralization_feature_indices: List[int],
    neutralization_proportion: float,
    neutralization_normalize: bool,
    tb: int,
) -> Optional[List[str]]:
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
            validation_stats,
            list(_decode(mix)),
            ensemble,
        )
        lut = {
            item[METRIC_PREDICTION_ID]: item[sort_by]
            for item in validation_stats
        }
        ranking = (
            pd.DataFrame(lut.items(), columns=['Prediction', sort_by])
            .set_index('Prediction')
            .sort_values(by=sort_by, ascending=sort_ascending)
            .to_markdown()
        )
        logger.info(f'Ranking:\n{ranking}')
    top = top_mix(lut, sort_ascending)
    if top:
        return list(_decode(top))
    return None
