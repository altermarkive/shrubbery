import sys
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import pandas as pd

from shrubbery.constants import COLUMN_ERA
from shrubbery.observability import logger

MODEL_SUBDIRECTORY = 'models'


def save_prediction(df: pd.DataFrame, name: str) -> Path:
    # Rank from 0 to 1 to meet diagnostic/submission file requirements
    stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    name = f'{stamp}_{name}'
    old = df.columns.to_list()[0]
    predictions = df.rank(pct=True).rename(columns={old: 'prediction'})
    predictions = predictions['prediction']
    with NamedTemporaryFile(
        prefix=f'{name}_', suffix='.csv', delete=False
    ) as prediction_file:
        prediction_path = Path(prediction_file.name)
    predictions.to_csv(prediction_path, index=True)
    return prediction_path


def store_model(model: Any, model_file: Path) -> None:
    pd.to_pickle(model, model_file, compression={'method': 'zip'})
    logger.info(f'Stored model: {model_to_string(model)}')


def load_model(model_file: Path) -> Any:
    if model_file.is_file():
        model = pd.read_pickle(model_file)
        logger.info(f'Loaded model: {model_to_string(model)}')
    else:
        logger.error('Model failed to materialize')
        return None
    return model


def pare_down_number_of_eras_in_training_data(
    training_data: pd.DataFrame, every_nth: int
) -> pd.DataFrame:
    every_nth_era = training_data[COLUMN_ERA].unique()[::every_nth]
    training_data = training_data[
        training_data[COLUMN_ERA].isin(every_nth_era)
    ]
    return training_data


def dict_of_lists_to_list_of_dicts(dict_of_lists: dict) -> list:
    return [
        dict(zip(dict_of_lists, list(item)))
        for item in zip(*dict_of_lists.values())
    ]


EPSILON = sys.float_info.epsilon


# Using rank takes care of the need to use this function
def trim_probability_array(array: np.ndarray) -> np.ndarray:
    return np.clip(array, 0.0 + EPSILON, 1.0 - EPSILON)


def identity(array: Any) -> Any:
    return array


def model_to_string(model: Any) -> str:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    model_name = model.__class__.__name__
    model_parameters = model.get_params(deep=False)
    description = f'{model_name}; {model_parameters}'
    return description.replace(' ', '').replace('\n', '')


class PrintableModelMixin:
    def __str__(self) -> str:
        return model_to_string(self)

    def __repr__(self) -> str:
        return model_to_string(self)
