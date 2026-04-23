import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import wandb

from .constants import COLUMN_ERA
from .observability import logger
from .workspace import get_workspace_path

MODEL_SUBDIRECTORY = 'models'
PREDICTIONS_SUBDIRECTORY = 'predictions'


def save_prediction(df: pd.DataFrame, name: str) -> Path:
    # Rank from 0 to 1 to meet diagnostic/submission file requirements
    stamp = datetime.now().strftime('%Y%m%d%H%M%S')
    name = f'{stamp}_{name}'
    old = df.columns.to_list()[0]
    predictions = df.rank(pct=True).rename(columns={old: 'prediction'})
    predictions = predictions['prediction']
    try:
        predictions_subdirectory = get_workspace_path(PREDICTIONS_SUBDIRECTORY)
    except Exception:
        logger.exception('Failed to locate workspace')
        pass
    prediction_path = predictions_subdirectory / f'{name}.csv'
    predictions.to_csv(prediction_path, index=True)
    return prediction_path


def store_model(model: Any, name: str) -> str:
    model_subdirectory = get_workspace_path(MODEL_SUBDIRECTORY)
    model_file = model_subdirectory / f'{name}.pkl.zip'
    pd.to_pickle(model, model_file, compression={'method': 'zip'})
    model_artifact = wandb.Artifact(name, type='model')
    model_artifact.add_file(str(model_file))
    version: str = 'latest'
    run = wandb.run
    if run is not None:
        model_artifact = run.log_artifact(model_artifact, aliases=[version])
        model_artifact.wait()
        version = model_artifact.version
    logger.info(f'Stored model: {model_to_string(model)}')
    return version


def load_model(name: str, version: str = 'latest') -> tuple[Any, str]:
    model_subdirectory = get_workspace_path(MODEL_SUBDIRECTORY)
    try:
        artifact = wandb.use_artifact(f'{name}:{version}', type='model')
        artifact.download(model_subdirectory)
        version = artifact.version
        logger.info(f'Downloaded model: {name}:{version}')
    except wandb.errors.CommError as exception:
        logger.error(f'W&B communication error: {exception}')
        return None, ''
    except requests.exceptions.HTTPError as exception:
        logger.error(f'HTTP communication error: {exception}')
        return None, ''
    model_file = model_subdirectory / f'{name}.pkl.zip'
    if model_file.is_file():
        model = pd.read_pickle(model_file)
        logger.info(f'Loaded model: {model_to_string(model)}')
    else:
        logger.error('Model failed to materialize')
        return None, ''
    return model, version


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
