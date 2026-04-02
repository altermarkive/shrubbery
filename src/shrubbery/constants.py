#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

os.environ['KERAS_BACKEND'] = 'torch'
from keras.utils import set_random_seed  # noqa: E402

COLUMN_ERA = 'era'
COLUMN_ID = 'id'
COLUMN_INDEX_ERA = 0
COLUMN_INDEX_TARGET = 0
COLUMN_INDEX_DATA_TYPE = -1
COLUMN_INDEX_EXAMPLE_PREDICTIONS = -1
COLUMN_TARGET = 'target'
COLUMN_DATA_TYPE = 'data_type'
COLUMN_DATA_TYPE_TRAINING = 'train'
COLUMN_DATA_TYPE_VALIDATION = 'validation'
COLUMN_DATA_TYPE_TOURNAMENT = 'live'
COLUMN_EXAMPLE_PREDICTIONS = 'example_predictions'
COLUMN_Y_PRED = 'y_pred'
COLUMN_Y_TRUE = 'y_true'
COLUMN_TRUE_CONTRIBUTION = 'tc'
COLUMN_ROUND_NUMBER = 'roundNumber'
COLUMN_MMC = 'mmc'
COLUMN_V2_CORR20 = 'v2_corr20'

RANDOM_SEED = 1337

set_random_seed(RANDOM_SEED)
