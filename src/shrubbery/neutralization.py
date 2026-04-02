from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
import scipy
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin

from shrubbery.constants import COLUMN_INDEX_ERA, COLUMN_INDEX_TARGET
from shrubbery.data.augmentation import get_biggest_change_features
from shrubbery.observability import logger
from shrubbery.utilities import PrintableModelMixin


def neutralize(
    x: np.ndarray,
    y: np.ndarray,
    neutralizers: Sequence[int],
    proportion: float,
    normalize: bool,
) -> np.ndarray:
    if not neutralizers:
        neutralizers = list(range(COLUMN_INDEX_ERA + 1, x.shape[1]))
    data = np.concatenate([x, y.reshape(-1, 1)], axis=1)
    eras = np.unique(x[:, COLUMN_INDEX_ERA])
    column_index_scores = data.shape[1] - 1
    computed = []
    for era in eras:
        data_era = data[data[:, COLUMN_INDEX_ERA] == era]
        scores = data_era[:, column_index_scores]
        if normalize:
            scores = (
                scipy.stats.rankdata(scores, method='ordinal') - 0.5
            ) / len(scores)
            scores = scipy.stats.norm.ppf(scores).reshape(-1, 1)
        exposures = data_era[:, neutralizers]
        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6).dot(
                scores.astype(np.float32)
            )
        )
        scores /= scores.std(ddof=0)
        computed.append(scores)
    return np.concatenate(computed)


def neutralize_series(
    series: pd.Series, by: pd.Series, proportion: float
) -> pd.Series:
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # This line makes series neutral to a constant column so that
    # it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (
            exposures,
            np.array([np.mean(series)] * len(exposures)).reshape(-1, 1),
        )
    )

    correction = proportion * exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]
    )
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized


class NumeraiNeutralization(
    BaseEstimator, MetaEstimatorMixin, RegressorMixin, PrintableModelMixin
):
    def __init__(
        self,
        estimator: Any,
        neutralization_cap: int,
        neutralization_proportion: float,
        neutralization_normalize: bool,
    ) -> None:
        self.estimator = estimator
        self.neutralization_cap = neutralization_cap
        self.neutralization_proportion = neutralization_proportion
        self.neutralization_normalize = neutralization_normalize

    def fit(
        self, x: np.ndarray, y: np.ndarray, **kwargs: Dict[str, Any]
    ) -> 'NumeraiNeutralization':
        riskiest_features = get_biggest_change_features(
            x,
            y[:, [COLUMN_INDEX_TARGET]],
            list(range(COLUMN_INDEX_ERA + 1, x.shape[1])),
            self.neutralization_cap,
        )
        self.neutralization_feature_indices_ = riskiest_features
        logger.info(f'Riskiest features (indices): {riskiest_features}')
        self.estimator = self.estimator.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        feature_indices = list(range(COLUMN_INDEX_ERA + 1, x.shape[1]))
        predictions = self.estimator.predict(
            x[:, feature_indices] if self.drop_era_column else x
        )
        neutralized = neutralize(
            x,
            predictions,
            self.neutralization_feature_indices_,
            self.neutralization_proportion,
            self.neutralization_normalize,
        )
        neutralized = np.concatenate(
            [
                x[:, COLUMN_INDEX_ERA].reshape(-1, 1),
                neutralized.reshape(-1, 1),
            ],
            axis=1,
        )
        # Ranking per era for all of our columns so we can combine safely
        # on the same scales.
        # See also:
        # https://forum.numer.ai/t/neutralization-output-in-5-5-range/6324
        predictions_columns = [1]
        predictions = (
            pd.DataFrame(neutralized)
            .groupby(COLUMN_INDEX_ERA, group_keys=False)
            .apply(
                lambda group: group[predictions_columns].rank(pct=True),
                include_groups=False,
            )
            .to_numpy()
        )
        return predictions
