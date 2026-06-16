from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin

from shrubbery.constants import COLUMN_INDEX_ERA
from shrubbery.utilities import PrintableModelMixin


class FitDownsamplerByEra(
    BaseEstimator, MetaEstimatorMixin, PrintableModelMixin
):
    def __init__(
        self, estimator: Any, era_stride: int, era_offset: int
    ) -> None:
        self.estimator = estimator
        self.era_stride = era_stride
        self.era_offset = era_offset

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        **kwargs: dict[str, Any],
    ) -> 'FitDownsamplerByEra':
        if self.era_stride > 1:
            eras = sorted(np.unique(x[:, COLUMN_INDEX_ERA]).tolist())
            eras_last_index = len(eras) - 1 - self.era_offset
            eras_first_index = eras_last_index % self.era_stride
            eras_downsampled = eras[eras_first_index :: self.era_stride]
            downsampled = np.isin(x[:, COLUMN_INDEX_ERA], eras_downsampled)
            result = self.estimator.fit(x[downsampled], y[downsampled], **kwargs)
        else:
            result = self.estimator.fit(x, y, **kwargs)
        self.fitted_ = True
        return result

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.estimator.predict(x)
