import time

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from shrubbery.constants import COLUMN_INDEX_ERA
from shrubbery.observability import logger
from shrubbery.utilities import model_to_string


class GenericEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimators: list,
        target_column_index: int,
    ) -> None:
        self.estimators = estimators
        self.target_column_index = target_column_index

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'GenericEmbedder':
        x_training = x[:, (COLUMN_INDEX_ERA + 1) :].astype(np.float32)
        y_training = y[:, self.target_column_index].astype(np.float32)
        for estimator in self.estimators:
            all_indices = list(range(x_training.shape[0]))
            indices = np.array(all_indices)
            embedder_string = model_to_string(estimator)
            logger.info(f'Running embedder training {embedder_string}')
            before = time.time()
            estimator.fit_transform(x_training[indices], y_training[indices])
            after = time.time()
            delta = int(after - before)
            logger.info(f'Completed embedder training in {delta} seconds')
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        eras = x[:, : (COLUMN_INDEX_ERA + 1)]
        features = x[:, (COLUMN_INDEX_ERA + 1) :].astype(np.float32)
        embeddings = []
        for estimator in self.estimators:
            embedder_string = model_to_string(estimator)
            logger.info(f'Running embedder transformation {embedder_string}')
            embedded = estimator.transform(features)
            if embedded.dtype == np.int64:
                embedded = embedded.astype(np.float32)
            embeddings.append(embedded)
        return np.concatenate([eras] + [features] + embeddings, axis=1)
