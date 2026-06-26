from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

PREDICT_ROW_CAP = 100000


class NumeraiPortionedPredict(BaseEstimator, RegressorMixin):
    # Chunks predict() into fixed-size row portions to cap peak memory.
    # CAUTION: chunking by a fixed row count splits eras across portions,
    # which corrupts any per-era neutralization and ranking (those are
    # global within an era). This wrapper must therefore sit INNER to the
    # neutralization estimator (it wraps the raw model and is itself wrapped
    # by Neutralization), so that neutralization still sees whole eras.
    def __init__(
        self, estimator: Any, predict_row_cap: int = PREDICT_ROW_CAP
    ) -> None:
        self.estimator = estimator
        self.predict_row_cap = predict_row_cap

    def fit(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict[str, Any]
    ) -> 'NumeraiPortionedPredict':
        self.estimator = self.estimator.fit(x, y, **kwargs)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        portions = [
            self.estimator.predict(x[start : start + self.predict_row_cap])
            for start in range(0, len(x), self.predict_row_cap)
        ]
        return np.concatenate(portions)
