#!/usr/bin/env python3

from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, MetaEstimatorMixin

from shrubbery.constants import COLUMN_INDEX_ERA
from shrubbery.utilities import PrintableModelMixin


class FitDownsamplerBySample(
    BaseEstimator, MetaEstimatorMixin, PrintableModelMixin
):
    def __init__(self, estimator: Any, sample_stride: int) -> None:
        self.estimator = estimator
        self.sample_stride = sample_stride

    def fit(
        self,
        x: NDArray,
        y: NDArray,
        **kwargs: Dict[str, Any],
    ) -> 'FitDownsamplerBySample':
        x = x[:: self.sample_stride]
        y = y[:: self.sample_stride]
        result = self.estimator.fit(x, y, **kwargs)
        self.fitted_ = True
        return result

    def predict(self, x: NDArray) -> NDArray:
        return self.estimator.predict(x)


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
        x: NDArray,
        y: NDArray,
        **kwargs: Dict[str, Any],
    ) -> 'FitDownsamplerByEra':
        eras = sorted(np.unique(x[:, COLUMN_INDEX_ERA]).tolist())
        eras_downsampled = eras[self.era_offset :: self.era_stride]
        downsampled = np.isin(x[:, COLUMN_INDEX_ERA], eras_downsampled)
        result = self.estimator.fit(x[downsampled], y[downsampled], **kwargs)
        self.fitted_ = True
        return result

    def predict(self, x: NDArray) -> NDArray:
        return self.estimator.predict(x)


class FitOnFeaturesOnly(
    BaseEstimator, MetaEstimatorMixin, PrintableModelMixin
):
    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(
        self,
        x: NDArray,
        y: NDArray,
        **kwargs: Dict[str, Any],
    ) -> 'FitOnFeaturesOnly':
        result = self.estimator.fit(x[:, COLUMN_INDEX_ERA:], y, **kwargs)
        self.fitted_ = True
        return result

    def predict(self, x: NDArray) -> NDArray:
        return self.estimator.predict(x[:, COLUMN_INDEX_ERA:])
