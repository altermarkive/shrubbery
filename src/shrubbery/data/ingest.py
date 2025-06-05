#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib
import json
import math
import random
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet

from shrubbery.constants import COLUMN_TARGET
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
    'validation_example_preds.parquet': {'investigate': False},
    'live_example_preds.parquet': {'investigate': False},
    'features.json': {'investigate': False},
}


def fully_qualify_file_name(
    directory: Path, file_name: str
) -> Tuple[Path, Path]:
    previous_file_name = file_name
    current_file_name = file_name
    return directory / current_file_name, directory / previous_file_name


def file_hash(file_path: Path) -> Optional[str]:
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
) -> Optional[bool]:
    different = None
    data_directory_path = get_workspace_path(DATA_SUBDIRECTORY)
    current_file_path, previous_file_path = fully_qualify_file_name(
        data_directory_path, file_name
    )
    logger.info(f'Downloading {file_name} to {current_file_path}')
    if investigate:
        previous_file_hash = file_hash(previous_file_path)
    napi.download_dataset(f'v5.0/{file_name}', str(current_file_path))
    if investigate:
        current_file_hash = file_hash(current_file_path)
        different = (
            current_file_hash != previous_file_hash
            if None not in [current_file_hash, previous_file_hash]
            else None
        )
    if file_name.endswith('.parquet'):
        schema = pyarrow.parquet.read_schema(
            current_file_path, memory_map=True
        )
        logger.info(f'Column count for {file_name}: {len(schema.names)}')
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
        file_to_download = FILES_TO_DOWNLOAD[file_name].copy()
        file_to_download['file_name'] = file_name
        different = download_file_and_investigate(**file_to_download)
        if different is not None and different:
            logger.warning(f'Warning - changed file content of {file_name}')


def locate_numerai_file(file_name: str) -> Path:
    data_directory_path = get_workspace_path(DATA_SUBDIRECTORY)
    file_path, _ = fully_qualify_file_name(data_directory_path, file_name)
    return file_path


def get_feature_set(selected_feature_set: str) -> List[str]:
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


def read_parquet_and_unpack_feature_encoding(
    file_name: str, read_columns: List[str], feature_cols: List[str]
) -> pd.DataFrame:
    data = pd.read_parquet(
        locate_numerai_file(file_name), columns=read_columns
    )
    # For more information about int8 encoding, see: https://forum.numer.ai/t/rain-data-release/6657  # noqa: E501
    data[feature_cols] = (
        data[feature_cols].apply(lambda x: x / 4.0).astype(np.float16)
    )
    return data


def _get_available_training_targets() -> Set[str]:
    train_parquet_file_path = locate_numerai_file('train.parquet')
    schema = pyarrow.parquet.read_schema(
        train_parquet_file_path, memory_map=True
    )
    available_targets = [
        column for column in schema.names if column.startswith('target')
    ]
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
    logger.info(f'Available targets - {sorted(finite_targets)}')
    return set(finite_targets)


def get_training_targets(k: int) -> List[str]:
    other_training_targets = _get_available_training_targets() - set(
        ['target', COLUMN_TARGET]
    )
    targets = [COLUMN_TARGET]
    if k > 0:
        # TODO: Randomly picking a handful of targets - this can be improved
        targets += sorted(random.sample(list(other_training_targets), k))
    logger.info(f'Selected targets - {targets}')
    return targets
