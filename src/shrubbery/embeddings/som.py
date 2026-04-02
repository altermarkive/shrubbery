import math

import numpy as np
import plotly.graph_objects as go
import wandb
from minisom import MiniSom
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from shrubbery.data.ingest import locate_numerai_file
from shrubbery.observability import logger


class SOM(BaseEstimator, TransformerMixin):
    def fit(self, x: np.ndarray, y: np.ndarray) -> 'SOM':
        som_grid_size = int(5 * math.sqrt(x.shape[1]))
        sigma = som_grid_size / (2 * math.sqrt(2))
        self.som_ = MiniSom(
            som_grid_size, som_grid_size, x.shape[1], sigma=sigma
        )
        self.som_.train(x, num_iteration=100, verbose=True)
        _plot_som_diagnostics(self.som_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.som_ is not None
        return _som_embed(self.som_, x)


def _som_embed(som: SOM, x: np.ndarray) -> np.ndarray:
    xy = np.array(
        [som.winner(sample) for sample in tqdm(x, desc='Embedding som')]
    )
    logger.info(f'SOM XY: {xy}')
    logger.info(f'SOM XY shape: {xy.shape}')
    return xy


def _plot_som_diagnostics(som: MiniSom) -> None:
    u_matrix = som.distance_map()
    fig = go.Figure(
        data=go.Heatmap(
            z=u_matrix, colorscale='Viridis', colorbar=dict(title='')
        )
    )
    fig.update_layout(
        title='U-Matrix',
        xaxis=dict(title='', showticklabels=False),
        yaxis=dict(title='', showticklabels=False, autorange='reversed'),
    )
    som_matrix_view_path = locate_numerai_file('som_matrix_view.png')
    fig.write_image(str(som_matrix_view_path))
    wandb.log({'SOM Matrix View': wandb.Image(str(som_matrix_view_path))})
