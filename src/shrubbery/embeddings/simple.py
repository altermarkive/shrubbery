#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from xgboost import XGBRegressor

from shrubbery.constants import (
    COLUMN_DATA_TYPE_TRAINING,
    COLUMN_INDEX_DATA_TYPE,
    COLUMN_INDEX_ERA,
)
from shrubbery.embeddings.diagnostics import plot_diagnostics
from shrubbery.observability import logger


def _apply_transform_in_chunks(
    estimator: Any, array: NDArray, chunk_size: int
) -> NDArray:
    if chunk_size <= 0:
        return estimator.transform(array)
    else:
        transformed = []
        for i in tqdm(range(math.ceil(array.shape[0] / chunk_size))):
            begin = i * chunk_size
            end = begin + chunk_size
            transformed.append(estimator.transform(array[begin:end]))
        return np.concatenate(transformed, axis=0)


@dataclass
class EmbedderConfig:
    name: str
    estimator: Any
    portion: float
    chunk_size: int
    verbosity: int
    dbscan_count: int
    dbscan_eps: float
    dbscan_min_samples: int


class GenericEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimators: List[EmbedderConfig],
        target_column_index: int,
    ) -> None:
        self.estimators = estimators
        self.target_column_index = target_column_index

    def fit(self, x: NDArray, y: NDArray) -> 'GenericEmbedder':
        training_data_selection = (
            x[:, COLUMN_INDEX_DATA_TYPE] == COLUMN_DATA_TYPE_TRAINING
        )
        x_training = x[
            training_data_selection,
            (COLUMN_INDEX_ERA + 1) : COLUMN_INDEX_DATA_TYPE,
        ].astype(np.float32)
        y_training = y[
            training_data_selection, self.target_column_index
        ].astype(np.float32)
        training_count = x_training.shape[0]
        for estimator in self.estimators:
            all_indices = list(range(x_training.shape[0]))
            if 0 < estimator.portion < 1:
                pick = int(x_training.shape[0] * estimator.portion)
                indices = np.array(random.sample(all_indices, pick))
            else:
                indices = np.array(all_indices)
            logger.info(f'Running embedder {estimator.name}')
            before = time.time()
            estimator.estimator.fit(x_training[indices])
            embeddings = estimator.estimator.transform(
                x_training[indices]
            )
            after = time.time()
            delta = int(after - before)
            logger.info(f'Completed embedder {estimator.name} in {delta}s')
            if estimator.verbosity > 0:
                plot_diagnostics(
                    estimator.name,
                    embeddings,
                    y_training,
                    training_count,
                    estimator.dbscan_count,
                    estimator.dbscan_eps,
                    estimator.dbscan_min_samples,
                    self.verbosity,
                )
        return self

    def transform(self, x: NDArray) -> NDArray:
        embeddings = []
        eras = x[:, : (COLUMN_INDEX_ERA + 1)]
        types = x[:, COLUMN_INDEX_DATA_TYPE:]
        for estimator in self.estimators:
            x_predicting = x[
                :, (COLUMN_INDEX_ERA + 1) : COLUMN_INDEX_DATA_TYPE
            ].astype(np.float32)
            embeddings.append(
                _apply_transform_in_chunks(
                    estimator.estimator, x_predicting, estimator.chunk_size
                )
            )
        return np.concatenate([eras] + embeddings + [types], axis=1)


class RandomizedSearchEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator: Any,
        params: Dict,
    ) -> None:
        self.estimator = estimator
        self.params = params

    def fit(self, x: NDArray, y: NDArray) -> 'RandomizedSearchEmbedder':
        pipeline = Pipeline(
            [
                ('embedder', self.estimator),
                (
                    'regressor',
                    XGBRegressor(device='cuda', tree_method='approx'),
                ),
            ]
        )
        params = {
            f'embedder__{key}': value for key, value in self.params.items()
        }
        search = RandomizedSearchCV(
            pipeline,
            params,
            n_iter=100,
            verbose=2,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
        )
        search.fit(x, y)
        logger.info(f'Best score for embedder is {search.best_score_}')
        logger.info(f'Best params for embedder are {search.best_params_}')
        self.estimator = search.best_estimator_.steps[0][1]
        return self

    def transform(self, x: NDArray) -> NDArray:
        return self.estimator.transform(x)
