#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from shrubbery.constants import (
    COLUMN_DATA_TYPE,
    COLUMN_DATA_TYPE_TOURNAMENT,
    COLUMN_DATA_TYPE_TRAINING,
    COLUMN_DATA_TYPE_VALIDATION,
    COLUMN_ERA,
    COLUMN_INDEX_ERA,
)
from shrubbery.data.ingest import locate_numerai_file
from shrubbery.observability import logger


def get_biggest_change_features(
    x: NDArray, y: NDArray, feature_indices: Sequence[int], n: int
) -> Sequence[int]:
    """
    Find the riskiest features by comparing their correlation vs
    the target in each split of training data
    (there are probably more clever ways to do this).

    Args:
        x (ndarray): Data to extract the riskiest features from
        y (ndarray): Primary target to correlate features with
        feature_indices (Sequence[int]): List of feature indices
        n (int): Specifies how many top riskiest features to return

    Returns:
        Sequence[int]: Riskiest features
    """
    logger.info(
        'Getting feature correlations over time and identifying riskiest ones'
    )
    all_eras = sorted(np.unique(x[:, COLUMN_INDEX_ERA]).tolist())
    h1_eras = all_eras[: len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2 :]

    data = np.concatenate([x, y.reshape(-1, 1)], axis=1)

    def per_era_feature_correlations(data: pd.DataFrame) -> pd.DataFrame:
        with warnings.catch_warnings():
            # For newly added features, some eras may be filled with 0.5
            # which results in a correlation of NaN hence we ignore the warning
            # and mean later on ignores NaN values (b/c dropna=True by default)
            warnings.filterwarnings(
                'ignore', message='invalid value encountered in divide'
            )
            # Getting the per era correlation of each feature
            # vs. the primary target across the training data (split)
            corrs = (
                pd.DataFrame(data)
                .groupby(COLUMN_INDEX_ERA, group_keys=False)
                .apply(
                    lambda era: era[feature_indices].corrwith(
                        era[era.shape[1] - 1]
                    ),
                    include_groups=False,
                )
            )
            return corrs

    h1_corr_means = per_era_feature_correlations(
        data[np.isin(data[:, COLUMN_INDEX_ERA], h1_eras)]
    ).mean()
    h2_corr_means = per_era_feature_correlations(
        data[np.isin(data[:, COLUMN_INDEX_ERA], h2_eras)]
    ).mean()

    corr_diffs = h2_corr_means - h1_corr_means
    worst_n = (
        corr_diffs.abs().sort_values(ascending=False).head(n).index.tolist()
    )

    return sorted(worst_n)


def numeric_eras(
    feature_cols: List[str],
    data: pd.DataFrame,
) -> Tuple[List[str], pd.DataFrame]:
    training_eras = sorted(
        list(
            set(
                data[data[COLUMN_DATA_TYPE] == COLUMN_DATA_TYPE_TRAINING][
                    COLUMN_ERA
                ]
            )
        )
    )
    logger.info(f'Eras (training): {training_eras[:2]} - {training_eras[-2:]}')
    validation_eras = sorted(
        list(
            set(
                data[data[COLUMN_DATA_TYPE] == COLUMN_DATA_TYPE_VALIDATION][
                    COLUMN_ERA
                ]
            )
        )
    )
    logger.info(
        f'Eras (validation): {validation_eras[:2]} - {validation_eras[-2:]}'
    )
    live_eras = sorted(
        list(
            set(
                data[data[COLUMN_DATA_TYPE] == COLUMN_DATA_TYPE_TOURNAMENT][
                    COLUMN_ERA
                ]
            )
        )
    )
    logger.info(f'Eras (live): {live_eras[:2]} - {live_eras[-2:]}')
    next_era = str(max([int(era) for era in validation_eras]) + 1)
    data.loc[
        data[COLUMN_DATA_TYPE] == COLUMN_DATA_TYPE_TOURNAMENT, COLUMN_ERA
    ] = next_era
    data[COLUMN_ERA] = data[COLUMN_ERA].astype(np.float32)
    return feature_cols, data


FILE_LIVE_IDS = 'live_ids.csv'
FILE_VALIDATION_IDS = 'validation_ids.csv'


def numerai_keep_ids(
    feature_cols: List[str],
    data: pd.DataFrame,
) -> Tuple[List[str], pd.DataFrame]:
    pd.DataFrame(
        data[data[COLUMN_DATA_TYPE] == COLUMN_DATA_TYPE_VALIDATION].index
    ).to_csv(locate_numerai_file(FILE_VALIDATION_IDS), index=False)
    pd.DataFrame(
        data[data[COLUMN_DATA_TYPE] == COLUMN_DATA_TYPE_TOURNAMENT].index
    ).to_csv(locate_numerai_file(FILE_LIVE_IDS), index=False)
    return feature_cols, data
