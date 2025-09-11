#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import plotly.express as px
import wandb
from numpy.typing import NDArray
from sklearn.cluster import DBSCAN


def _plot_2d_scatter(
    content: str, name: str, coordinates: NDArray, colors: NDArray
) -> None:
    title = f't-SNE ({content}): {name}'
    df = pd.DataFrame(
        np.concatenate([coordinates, colors], axis=1),
        columns=['x', 'y', 'color'],
    )
    df['color'] = df['color'].astype(str)
    fig = px.scatter(df, x='x', y='y', color='color')
    fig.update_traces(marker=dict(size=2))
    wandb.log({title: fig})


def _plot_3d_scatter(
    content: str, name: str, coordinates: NDArray, colors: NDArray
) -> None:
    title = f't-SNE ({content}): {name}'
    df = pd.DataFrame(
        np.concatenate([coordinates, colors], axis=1),
        columns=['x', 'y', 'z', 'color'],
    )
    df['color'] = df['color'].astype(str)
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='color')
    fig.update_traces(marker=dict(size=2))
    wandb.log({title: fig})


def _plot_data(
    content: str, name: str, coordinates: NDArray, colors: NDArray
) -> None:
    if coordinates.shape[1] == 2:
        _plot_2d_scatter(
            content,
            name,
            coordinates,
            colors,
        )
    elif coordinates.shape[1] == 3:
        _plot_3d_scatter(
            content,
            name,
            coordinates,
            colors,
        )


def _plot_reference(
    name: str, embeddings: NDArray, targets: pd.DataFrame, training_count: int
) -> None:
    _plot_data(
        'reference',
        name,
        embeddings[:training_count],
        targets[:training_count],
    )


def _plot_output(
    name: str,
    embeddings: NDArray,
    dbscan_count: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
) -> None:
    idx = np.random.choice(len(embeddings), dbscan_count)
    dbscan = DBSCAN(
        eps=dbscan_eps,  # This appears to be related to the number of samples
        min_samples=dbscan_min_samples,
    )
    dbscan_all = np.reshape(dbscan.fit_predict(embeddings[idx]), (-1, 1))
    _plot_data(
        'output',
        name,
        embeddings[idx],
        dbscan_all,
    )


def plot_diagnostics(
    name: str,
    embeddings: NDArray,
    targets: NDArray,
    training_count: int,
    dbscan_count: int,
    dbscan_eps: float,
    dbscan_min_samples: int,
    verbosity: int,
) -> None:
    if verbosity > 0:
        if verbosity > 1:
            _plot_reference(name, embeddings, targets, training_count)
        _plot_output(
            name, embeddings, dbscan_count, dbscan_eps, dbscan_min_samples
        )
