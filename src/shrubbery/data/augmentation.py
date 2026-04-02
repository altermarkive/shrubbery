import warnings

import numpy as np
import pandas as pd

from shrubbery.constants import (
    COLUMN_ERA,
    COLUMN_INDEX_ERA,
)
from shrubbery.observability import logger


def get_biggest_change_features(
    x: np.ndarray, y: np.ndarray, feature_indices: list[int], n: int
) -> list[int]:
    """
    Find the riskiest features by comparing their correlation vs
    the target in each split of training data
    (there are probably more clever ways to do this).

    Args:
        x (np.ndarray): Data to extract the riskiest features from
        y (np.ndarray): Primary target to correlate features with
        feature_indices (list[int]): List of feature indices
        n (int): Specifies how many top riskiest features to return

    Returns:
        list[int]: Riskiest features
    """
    logger.info(
        'Getting feature correlations over time and identifying riskiest ones'
    )
    all_eras = sorted(np.unique(x[:, COLUMN_INDEX_ERA]).tolist())
    h1_eras = all_eras[: len(all_eras) // 2]
    h2_eras = all_eras[len(all_eras) // 2 :]

    data = np.concatenate([x, y.reshape(-1, 1)], axis=1)

    def per_era_feature_correlations(data: np.ndarray) -> pd.DataFrame:
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


def numeric_eras(name: str, data: pd.DataFrame) -> list:
    eras = sorted(list(set(data[COLUMN_ERA])))
    logger.info(f'Eras in {name}: {eras[:2]} - {eras[-2:]}')
    return eras


def override_numerai_era(eras: list, data: pd.DataFrame) -> None:
    next_era = np.float32(max([int(era) for era in eras]) + 1)
    data[COLUMN_ERA] = next_era
