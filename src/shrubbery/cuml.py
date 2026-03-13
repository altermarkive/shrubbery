from typing import Any

import cupy
import numpy as np
import rmm
from sklearn.base import BaseEstimator, RegressorMixin


class CumlIsolationRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        estimator: Any,
    ):
        self.estimator = estimator
        self.is_fitted_ = False

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'CumlIsolationRegressor':
        rmm.reinitialize(pool_allocator=False)  # Use a non-pooling allocator
        self.estimator.fit(x, y)
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        self.is_fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        rmm.reinitialize(pool_allocator=False)  # Use a non-pooling allocator
        predictions = self.estimator.predict(x)
        cupy.get_default_memory_pool().free_all_blocks()
        cupy.get_default_pinned_memory_pool().free_all_blocks()
        return predictions
