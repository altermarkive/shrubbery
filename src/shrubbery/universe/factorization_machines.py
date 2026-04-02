#!/usr/bin/env python3

import numpy as np
from myfm import MyFMRegressor
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin


class FMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        rank: int,
        init_stdev: float,
        random_seed: int,
        alpha_0: float,
        beta_0: float,
        gamma_0: float,
        mu_0: float,
        reg_0: float,
        fit_w0: bool,
        fit_linear: bool,
    ) -> None:
        self.rank = rank
        self.init_stdev = init_stdev
        self.random_seed = random_seed
        self.alpha_0 = alpha_0
        self.beta_0 = beta_0
        self.gamma_0 = gamma_0
        self.mu_0 = mu_0
        self.reg_0 = reg_0
        self.fit_w0 = fit_w0
        self.fit_linear = fit_linear

    def fit(self, x: NDArray, y: NDArray) -> 'FMRegressor':
        self.model_ = MyFMRegressor(**self.get_params())
        self.model_ = self.model_.fit(
            x.astype(np.float32), y.astype(np.float32)
        )
        return self

    def predict(self, x: NDArray) -> NDArray:
        assert self.model_ is not None
        return self.model_.predict(x.astype(np.float32))
