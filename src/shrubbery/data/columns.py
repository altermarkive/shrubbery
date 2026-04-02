from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.data.ingest import lookup_target_index

from shrubbery.constants import COLUMN_INDEX_ERA


class NumeraiFeaturesSelector(ColumnTransformer):
    def __init__(self) -> None:
        super().__init__(
            transformer='drop',
            columns=[COLUMN_INDEX_ERA],
            remainder='passthrough',
        )


class NumeraiTargetSelector(
    BaseEstimator, MetaEstimatorMixin, TransformerMixin
):
    def __init__(self, estimator: Any, target: int | str) -> None:
        self.estimator = estimator
        self.target = target

    def fit(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict[str, Any]
    ) -> 'NumeraiTargetSelector':
        if isinstance(self.target, int):
            target = self.target
        else:
            target = lookup_target_index(self.target)
        self.estimator.fit(x, y[[target]].ravel())
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict(x)
