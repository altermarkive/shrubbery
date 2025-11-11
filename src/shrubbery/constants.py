#!/usr/bin/env python3

import os

os.environ['KERAS_BACKEND'] = 'torch'
from keras.utils import set_random_seed  # noqa: E402

COLUMN_ERA = 'era'
COLUMN_ID = 'id'
COLUMN_INDEX_ERA = 0
COLUMN_INDEX_TARGET = 0
COLUMN_TARGET = 'target'
COLUMN_Y_PRED = 'y_pred'
COLUMN_Y_TRUE = 'y_true'
COLUMN_TRUE_CONTRIBUTION = 'tc'
COLUMN_ROUND_NUMBER = 'roundNumber'
COLUMN_MMC = 'mmc'
COLUMN_V2_CORR20 = 'v2_corr20'

RANDOM_SEED = 1337

set_random_seed(RANDOM_SEED)
