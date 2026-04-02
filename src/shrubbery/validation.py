import operator
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.model_selection import BaseCrossValidator

from shrubbery.constants import COLUMN_INDEX_ERA
from shrubbery.data.ingest import locate_numerai_file
from shrubbery.observability import logger
from shrubbery.utilities import dict_of_lists_to_list_of_dicts


# Numerai-specific cross-validation generator
# See also:
# * https://scikit-learn.org/stable/glossary.html#term-CV-splitter
# * https://scikit-learn.org/stable/modules/cross_validation.html
# * https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b/sklearn/model_selection/_split.py#L469  # noqa: E501
class NumeraiTimeSeriesSplitter(BaseCrossValidator):
    def __init__(self, cv: int, embargo: int) -> None:
        self._cv = cv
        self._embargo = embargo

    def get_n_splits(
        self, x: np.ndarray, y: np.ndarray, groups: Any = None
    ) -> int:  # ty: ignore[invalid-method-override]
        # See also:
        # - https://scikit-learn.org/stable/glossary.html#term-get_n_splits
        return self._cv

    def split(
        self, x: np.ndarray, y: np.ndarray, groups: Any = None
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:  # ty: ignore[invalid-method-override]
        # See also:
        # - https://scikit-learn.org/stable/glossary.html#term-split
        assert x is not None, 'Training vector was not provided'
        assert y is not None, 'Target values were not provided'
        assert x.shape[0] == y.shape[0], 'Mismatch of input shapes'

        eras = x[:, COLUMN_INDEX_ERA]
        all_train_eras = np.unique(eras)
        len_split = len(all_train_eras) // self._cv
        test_splits = [
            all_train_eras[i * len_split : (i + 1) * len_split]
            for i in range(self._cv)
        ]
        # Fix the last test split to have all the last eras,
        # in case the number of eras wasn't divisible by cv
        remainder = len(all_train_eras) % self._cv
        if remainder != 0:
            test_splits[-1] = np.append(
                test_splits[-1], all_train_eras[-remainder:]
            )

        train_splits = []
        for test_split in test_splits:
            test_split_max = int(np.max(test_split))
            test_split_min = int(np.min(test_split))
            # Get all of the eras that aren't in the test split
            train_split_not_embargoed = [
                e
                for e in all_train_eras
                if not (test_split_min <= int(e) <= test_split_max)
            ]
            # Embargo the train split so we have no leakage.
            # one era is length 5, so we need to embargo
            # by target_length/5 eras.
            # To be consistent for all targets, let's embargo everything
            # by 60/5 == 12 eras.
            train_split = [
                e
                for e in train_split_not_embargoed
                if abs(int(e) - test_split_max) > self._embargo
                and abs(int(e) - test_split_min) > self._embargo
            ]
            train_splits.append(train_split)

        # Convenient way to iterate over train and test splits
        training_data = pd.DataFrame(x)
        train_test_zip = zip(train_splits, test_splits)
        for train_test_split in train_test_zip:
            train_split, test_split = train_test_split
            train_split_index = training_data[COLUMN_INDEX_ERA].isin(
                train_split
            )
            test_split_index = training_data[COLUMN_INDEX_ERA].isin(test_split)
            train_split_shape = train_split_index[train_split_index].shape
            test_split_shape = test_split_index[test_split_index].shape
            logger.info(f'Train split shape: {train_split_shape}')
            logger.info(f'Test split shape: {test_split_shape}')
            # Using `.copy(deep=True)` for better performance
            train_split_array = train_split_index.copy(deep=True).to_numpy(
                dtype=np.bool_
            )
            test_split_array = test_split_index.copy(deep=True).to_numpy(
                dtype=np.bool_
            )
            yield train_split_array, test_split_array


def reformat_cross_validation_result(
    cross_validation_result: dict, model_name: str
) -> list[float]:
    results = sorted(
        dict_of_lists_to_list_of_dicts(cross_validation_result),
        key=operator.itemgetter('rank_test_score'),
    )
    cross_validation_table = wandb.Table(data=pd.DataFrame(results))
    cross_validation_table_name = f'Cross-validation Table ({model_name})'
    wandb.log({cross_validation_table_name: cross_validation_table})
    return results


def get_best_parameters(
    results: list, parameter: str, top_k: int
) -> list[str]:
    best_parameters = [item['params'][parameter] for item in results[:top_k]]
    logger.info(
        f'Best cross-validation values for "{parameter}": {best_parameters}'
    )
    return best_parameters


def cross_validation_to_parallel_coordinates(
    cross_validation_result: dict, model_name: str
) -> None:
    result = pd.DataFrame(cross_validation_result)
    columns = [
        column for column in result.columns if column.startswith('param_')
    ] + ['mean_test_score']
    result = result[columns]
    result.columns = [
        column.replace('param_', '') for column in result.columns
    ]
    for column in result.columns:
        if result[column].dtype == 'object':
            values = result[column].apply(str)
            categories = values.unique().tolist()
            result[column] = values.apply(categories.index)
    result = result.sort_values('mean_test_score').reset_index(drop=True)
    data_columns = list(result.columns)
    column_mins = result.min()
    column_maxs = result.max()
    column_range = column_maxs - column_mins
    result = (result - column_mins) / column_range
    figure, axes = plt.subplots(figsize=(12, 6))
    cmap = plt.colormaps['plasma']
    norm = plt.Normalize(
        vmin=column_mins['mean_test_score'],
        vmax=column_maxs['mean_test_score'],
    )
    x = range(len(data_columns))
    for _, row in result.iterrows():
        axes.plot(
            x, row[data_columns], color=cmap(row['mean_test_score']), alpha=0.7
        )
    for i in x:
        axes.axvline(i, color='black', linewidth=0.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=axes)
    for i, column in enumerate(data_columns):
        axes.text(
            i + 0.02,
            1.02,
            f'{column_maxs[column]:.3g}',
            ha='left',
            va='bottom',
            fontsize=8,
        )
        axes.text(
            i + 0.02,
            -0.02,
            f'{column_mins[column]:.3g}',
            ha='left',
            va='top',
            fontsize=8,
        )
        axes.text(i, -0.06, column, ha='center', va='top', fontsize=8)
    axes.set_yticks([])
    axes.set_xticks([])
    title = f'Cross-validation result for {model_name}'
    axes.set_title(title)
    plt.tight_layout()
    plot_path = locate_numerai_file(f'cross_validation_{model_name}.png')
    plt.savefig(plot_path)
    wandb.log({title: wandb.Image(str(plot_path))})
    plt.close(figure)
