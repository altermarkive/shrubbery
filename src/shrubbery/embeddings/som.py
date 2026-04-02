#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from shrubbery.data.ingest import locate_numerai_file
from shrubbery.observability import logger

import wandb  # isort: skip


class SOM(BaseEstimator, TransformerMixin):
    def fit(self, x: NDArray, y: NDArray) -> 'SOM':
        som_grid_size = int(5 * math.sqrt(x.shape[1]))
        sigma = som_grid_size / (2 * math.sqrt(2))
        self.som_ = MiniSom(
            som_grid_size, som_grid_size, x.shape[1], sigma=sigma
        )
        self.som_.train(x, num_iteration=100, verbose=True)
        _plot_som_diagnostics(self.som_)
        return self

    def transform(self, x: NDArray) -> NDArray:
        assert self.som_ is not None
        return _som_embed(self.som_, x)


def _som_embed(som: SOM, x: NDArray) -> NDArray:
    xy = np.array(
        [som.winner(sample) for sample in tqdm(x, desc='Embedding som')]
    )
    logger.info(f'SOM XY: {xy}')
    logger.info(f'SOM XY shape: {xy.shape}')
    return xy


def _plot_som_diagnostics(som: MiniSom) -> None:
    u_matrix = som.distance_map()
    plt.imshow(u_matrix, cmap='viridis')
    plt.colorbar()
    plt.title('U-Matrix')
    plt.show()
    som_matrix_view_path = locate_numerai_file('som_matrix_view.png')
    plt.savefig(som_matrix_view_path)
    wandb.log({'SOM Matrix View': wandb.Image(str(som_matrix_view_path))})
