#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Sequence

import numpy as np
import pandas as pd
import scipy
from numpy.typing import NDArray

from shrubbery.constants import COLUMN_INDEX_ERA


def neutralize(
    x: NDArray,
    y: NDArray,
    neutralizers: Sequence[int],
    proportion: float,
    normalize: bool,
) -> NDArray:
    if not neutralizers:
        neutralizers = list(range(COLUMN_INDEX_ERA + 1, x.shape[1]))
    data = np.concatenate([x, y.reshape(-1, 1)], axis=1)
    eras = np.unique(x[:, COLUMN_INDEX_ERA])
    column_index_scores = data.shape[1] - 1
    computed = []
    for era in eras:
        data_era = data[data[:, COLUMN_INDEX_ERA] == era]
        scores = data_era[:, column_index_scores]
        if normalize:
            scores = (
                scipy.stats.rankdata(scores, method='ordinal') - 0.5
            ) / len(scores)
            scores = scipy.stats.norm.ppf(scores).reshape(-1, 1)
        exposures = data_era[:, neutralizers]
        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32), rcond=1e-6).dot(
                scores.astype(np.float32)
            )
        )
        scores /= scores.std(ddof=0)
        computed.append(scores)
    return np.concatenate(computed)


def neutralize_series(
    series: pd.Series, by: pd.Series, proportion: float
) -> pd.Series:
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # This line makes series neutral to a constant column so that
    # it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (
            exposures,
            np.array([np.mean(series)] * len(exposures)).reshape(-1, 1),
        )
    )

    correction = proportion * exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]
    )
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized
