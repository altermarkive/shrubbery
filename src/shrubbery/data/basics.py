import numpy as np
import pandas as pd
import plotly.express as px
import wandb
from sklearn.base import BaseEstimator, TransformerMixin

from shrubbery.constants import (
    COLUMN_INDEX_ERA,
    COLUMN_INDEX_TARGET,
)


class PlotBasicDataInformation(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        return

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'PlotBasicDataInformation':
        not_nan_target_data_selection = ~np.isnan(y[:, COLUMN_INDEX_TARGET])
        selection = not_nan_target_data_selection
        x = x[selection, (COLUMN_INDEX_ERA + 1) :]
        y = y[selection, COLUMN_INDEX_TARGET].reshape(-1, 1)
        _plot_correlation_heatmap('Feature Correlation', x)
        _plot_distribution_violins('Feature Distributions', x)
        _plot_distribution_scatter_matrix('Feature Scatter Matrix', x, y)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x


def _plot_correlation_heatmap(title: str, x: np.ndarray) -> None:
    features = pd.DataFrame(x)
    correlation_matrix = features.corr()
    fig = px.imshow(correlation_matrix, color_continuous_scale='RdBu_r')
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_coloraxes(showscale=False)
    wandb.log({title: fig})


def _plot_distribution_violins(title: str, x: np.ndarray) -> None:
    features = pd.DataFrame(x).sample(n=10000)
    fig = px.violin(features, box=True)
    wandb.log({title: fig})


def _plot_distribution_scatter_matrix(
    title: str, x: np.ndarray, y: np.ndarray
) -> None:
    features = pd.DataFrame(np.concatenate([x, y], axis=1)).sample(n=1000)
    fig = px.scatter_matrix(
        features,
        dimensions=list(range(x.shape[1])),
        color=COLUMN_INDEX_TARGET,
        size_max=1,
    )
    fig.update_traces(diagonal_visible=False, marker=dict(size=2))
    config = dict(range=[0, 1], visible=False, showticklabels=False)
    fig.update_layout({f'yaxis{i + 1}': config for i in range(x.shape[1])})
    fig.update_layout({f'xaxis{i + 1}': config for i in range(x.shape[1])})
    wandb.log({title: fig})
