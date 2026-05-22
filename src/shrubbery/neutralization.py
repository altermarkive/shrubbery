from typing import Any

import numpy as np
import pandas as pd
import scipy
from numerai_tools.scoring import neutralize as numerai_tools_neutralize
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin

from shrubbery.constants import (
    COLUMN_ERA,
    COLUMN_FEATURE_PREFIX,
    COLUMN_INDEX_ERA,
    COLUMN_INDEX_TARGET,
    COLUMN_TARGET,
)
from shrubbery.data.augmentation import get_biggest_change_features
from shrubbery.observability import logger
from shrubbery.utilities import PrintableModelMixin


def neutralize(
    x: np.ndarray,
    y: np.ndarray,
    neutralizers: list[int],
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


class NumeraiNeutralization(
    BaseEstimator, MetaEstimatorMixin, RegressorMixin, PrintableModelMixin
):
    def __init__(
        self,
        estimator: Any,
        neutralization_cap: int | None,
        neutralization_proportion: float,
        neutralization_normalize: bool,
    ) -> None:
        self.estimator = estimator
        self.neutralization_cap = neutralization_cap
        self.neutralization_proportion = neutralization_proportion
        self.neutralization_normalize = neutralization_normalize

    def fit(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict[str, Any]
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
        predictions = self.estimator.predict(x)
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


def _combine_x_y_into_frame(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    feature_count = x.shape[1] - COLUMN_INDEX_ERA - 1
    feature_columns = [
        f'{COLUMN_FEATURE_PREFIX}_{index}' for index in range(feature_count)
    ]
    frame = pd.DataFrame(x, columns=pd.Index([COLUMN_ERA] + feature_columns))
    frame[COLUMN_TARGET] = np.asarray(y).reshape(-1)
    return frame


class NumeraiToolsNeutralization(
    BaseEstimator, MetaEstimatorMixin, RegressorMixin, PrintableModelMixin
):
    def __init__(
        self,
        estimator: Any,
        neutralization_proportion: float,
    ) -> None:
        self.estimator = estimator
        self.neutralization_proportion = neutralization_proportion

    def fit(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict[str, Any]
    ) -> 'NumeraiToolsNeutralization':
        self.estimator = self.estimator.fit(x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = self.estimator.predict(x)
        frame = _combine_x_y_into_frame(x, predictions)
        feature_columns = [
            column
            for column in frame.columns
            if column.startswith(COLUMN_FEATURE_PREFIX)
        ]
        neutralized = frame.groupby(COLUMN_ERA, group_keys=False).apply(
            lambda group: numerai_tools_neutralize(
                group[[COLUMN_TARGET]].copy(),
                group[feature_columns].copy(),
                self.neutralization_proportion,
            ),
            include_groups=False,
        )
        frame[COLUMN_TARGET] = neutralized[COLUMN_TARGET]
        # Ranking per era so columns combine safely on the same scales.
        # https://forum.numer.ai/t/neutralization-output-in-5-5-range/6324
        ranked = frame.groupby(COLUMN_ERA, group_keys=False)[
            COLUMN_TARGET
        ].rank(pct=True)
        return ranked.to_numpy().reshape(-1, 1)
