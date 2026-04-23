import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from shrubbery.constants import COLUMN_ERA
from shrubbery.data.augmentation import numeric_eras
from shrubbery.napi import napi
from shrubbery.observability import logger
from shrubbery.workspace import get_workspace_path

DATA_SUBDIRECTORY = 'data'


# Tournament data changes every week so we specify the round in their name.
# Training and validation data only change periodically, so no need to
# download them every time.
FILES_TO_DOWNLOAD = {
    'train.parquet': {'investigate': True},
    'validation.parquet': {'investigate': False},
    'live.parquet': {'investigate': False},
    'features.json': {'investigate': False},
}


def fully_qualify_file_name(
    directory: Path, file_name: str
) -> tuple[Path, Path]:
    previous_file_name = file_name
    current_file_name = file_name
    return directory / current_file_name, directory / previous_file_name


def file_hash(file_path: Path) -> str | None:
    try:
        with open(file_path, 'rb') as handle:
            file_hash_object = hashlib.sha512()
            while chunk := handle.read(1048576):
                file_hash_object.update(chunk)
            return file_hash_object.hexdigest()
    except Exception:
        return None


def download_file_and_investigate(
    file_name: str,
    investigate: bool = False,
) -> bool | None:
    different = None
    data_directory_path = get_workspace_path(DATA_SUBDIRECTORY)
    current_file_path, previous_file_path = fully_qualify_file_name(
        data_directory_path, file_name
    )
    logger.info(f'Downloading {file_name} to {current_file_path}')
    if investigate:
        previous_file_hash = file_hash(previous_file_path)
    napi.download_dataset(f'v5.2/{file_name}', str(current_file_path))
    if investigate:
        current_file_hash = file_hash(current_file_path)
        different = (
            current_file_hash != previous_file_hash
            if None not in [current_file_hash, previous_file_hash]
            else None
        )
    return different


def log_all_dataset_entries() -> None:
    file_names = napi.list_datasets()
    digits = math.ceil(math.log10(len(file_names) + 1))
    for i, file_name in enumerate(file_names):
        logger.info(f'Dataset list entry #{str(i).zfill(digits)}: {file_name}')


def download_numerai_files():
    log_all_dataset_entries()
    logger.info('Downloading dataset files...')
    for file_name in FILES_TO_DOWNLOAD:
        investigate = FILES_TO_DOWNLOAD[file_name]['investigate']
        different = download_file_and_investigate(file_name, investigate)
        if different is not None and different:
            logger.warning(f'Warning - changed file content of {file_name}')


def locate_numerai_file(file_name: str) -> Path:
    data_directory_path = get_workspace_path(DATA_SUBDIRECTORY)
    file_path, _ = fully_qualify_file_name(data_directory_path, file_name)
    return file_path


def get_feature_set(selected_feature_set: str) -> list[str]:
    # Read the feature metadata and get a feature set
    with open(locate_numerai_file('features.json'), 'r') as handle:
        feature_metadata = json.load(handle)
    all_features = set()
    for feature_set in feature_metadata['feature_sets'].values():
        all_features.update(feature_set)
    feature_count = len(all_features)
    logger.info(f'Feature count - all: {feature_count}')
    for feature_set in feature_metadata['feature_sets'].keys():
        feature_set_length = len(feature_metadata['feature_sets'][feature_set])
        logger.info(f'Feature count - {feature_set}: {feature_set_length}')
    features = feature_metadata['feature_sets'][selected_feature_set]
    return sorted(features)


def read_parquet_and_unpack(
    file_name: str, read_columns: list[str], feature_cols: list[str]
) -> tuple[pd.DataFrame, list]:
    logger.info(f'Reading {file_name}')
    data = pd.read_parquet(
        locate_numerai_file(file_name), columns=read_columns
    )
    # For more information about int8 encoding, see: https://forum.numer.ai/t/rain-data-release/6657  # noqa: E501
    data[feature_cols] = (
        data[feature_cols].apply(lambda x: x / 4.0).astype(np.float16)
    )
    data_column_era = data[COLUMN_ERA]
    data_column_era = np.where(data_column_era == 'X', np.nan, data_column_era)
    data[COLUMN_ERA] = data_column_era.astype(np.float32)
    eras = numeric_eras(file_name, data)
    return data, eras


def get_training_targets() -> list[str]:
    with open(locate_numerai_file('features.json'), 'r') as handle:
        feature_metadata = json.load(handle)
    available_targets = feature_metadata['targets']
    train_parquet_file_path = locate_numerai_file('train.parquet')
    training_data = pd.read_parquet(
        train_parquet_file_path, columns=available_targets
    )
    finite_targets = [
        target
        for target in available_targets
        if not (
            np.isinf(training_data[target]).any()
            or np.isnan(training_data[target]).any()
        )
    ]
    for target in available_targets:
        if target not in finite_targets:
            logger.warning(
                f'Dropped {target} after checking for infinity & NaN'
            )
    finite_targets = sorted(finite_targets)
    target_names = locate_numerai_file('target_names.json')
    with target_names.open('w') as handle:
        json.dump(finite_targets, handle, indent=4)
    logger.info(f'Targets - {finite_targets}')
    return finite_targets


def lookup_target_index(target_name: str) -> int:
    target_names = locate_numerai_file('target_names.json')
    with target_names.open('r') as handle:
        training_targets = json.load(handle)
    return training_targets.index(target_name)
