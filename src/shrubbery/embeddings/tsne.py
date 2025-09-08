#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

from numpy.typing import NDArray
from openTSNE import TSNE
from sklearn.base import BaseEstimator, TransformerMixin


class OpenTSNE(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator: TSNE,
    ) -> None:
        self.estimator = estimator

    def fit(self, x: NDArray, y: NDArray) -> 'OpenTSNE':
        self.embedder_ = self.estimator.fit(x)
        return self

    def transform(self, x: NDArray) -> NDArray:
        assert self.embedder_ is not None
        return self.embedder_.transform(x)
