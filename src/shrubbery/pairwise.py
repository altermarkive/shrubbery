import time
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

from shrubbery.observability import logger


class Pairwise(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    def __init__(
        self,
        estimator: Any,
        n_shuffles_fit: int,
        n_shuffles_predict: int,
    ) -> None:
        self.estimator = estimator
        self.n_shuffles_fit = n_shuffles_fit
        self.n_shuffles_predict = n_shuffles_predict

    def fit(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict[str, Any]
    ) -> 'Pairwise':
        y = y.reshape(-1, 1)
        logger.info('Fitting...')
        joint_x_list = []
        joint_y_list = []
        for _ in range(self.n_shuffles_fit):
            shuffled_x, shuffled_y = shuffle(x, y)
            joint_x = np.concatenate([x, shuffled_x], axis=1)
            selected = (y != shuffled_y).ravel()
            joint_x = joint_x[selected]
            joint_y = y[selected]
            joint_x_list.append(joint_x)
            joint_y_list.append(joint_y)
        joint_x = np.concatenate(joint_x_list)
        joint_y = np.concatenate(joint_y_list).ravel()
        start_time = time.time()
        self.estimator = self.estimator.fit(joint_x, joint_y)
        stop_time = time.time()
        logger.info(f'Fitted in {int(stop_time - start_time)}s')
        mse = mean_squared_error(joint_y, self.estimator.predict(joint_x))
        logger.info(f'MSE: {mse}')
        self.fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        predictions = []
        for _ in range(self.n_shuffles_predict):
            shuffled_x = shuffle(x)
            joint_x = np.concatenate([x, shuffled_x], axis=1)
            predictions.append(self.estimator.predict(joint_x))
        return np.mean(np.array(predictions), axis=0)
