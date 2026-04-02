import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from shrubbery.constants import COLUMN_ERA, RANDOM_SEED
from shrubbery.observability import logger

COLUMN_DATA_TYPE_REFERENCE = 'type_reference'
COLUMN_DATA_TYPE_CONFIDENCE = 'type_confidence'

TRAINING_DATA = 0
VALIDATION_DATA = 1


# In general, the idea is to redivide training set
# according to correlation to validation set
# to avoid overfitting.
# Similar to: https://github.com/altermarkive/numerai-experiments/blob/master/src/ml-jimfleming--numerai/prep_data.py  # noqa
# Which was described here: https://medium.com/jim-fleming/notes-on-the-numerai-ml-competition-14e3d42c19f3  # noqa
# And based on the method described here: http://fastml.com/adversarial-validation-part-one/  # noqa
# In comparison, this code uses XGBClassifier to cope
# with a larger volume of data, not using StratifiedKFold
# because it was invariably producing folds with just one class
# with current training/validation ratio and using only the part
# of the training dataset most similar to validation dataset
# (and not creating a separate validation dataset
# out of the least similar portion of training dataset).
# Adversarial validation was made era-aware by applying the concept
# within eras rather than to individual samples.
def adversarial_downsampling(
    feature_cols: list[str],
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    downsampling_ratio: float,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not (0.0 < downsampling_ratio < 1.0):
        return feature_cols, training_data, validation_data

    training_data[COLUMN_DATA_TYPE_REFERENCE] = TRAINING_DATA
    validation_data[COLUMN_DATA_TYPE_REFERENCE] = VALIDATION_DATA

    data = pd.concat([training_data, validation_data])

    x_split = data[[COLUMN_ERA] + feature_cols]
    y_split = data[[COLUMN_DATA_TYPE_REFERENCE]]

    classifier = XGBClassifier(
        n_estimators=100,
        verbosity=1,
        n_jobs=-1,
        random_state=RANDOM_SEED,
        device='cuda',
        tree_method='approx',
    )
    results = pd.concat(
        [
            x_split[[COLUMN_ERA]],
            y_split,
            pd.DataFrame(
                np.zeros(y_split.size),
                index=data.index,
                columns=[COLUMN_DATA_TYPE_CONFIDENCE],
            ),
        ],
        axis=1,
    )

    logger.info('Adversarial Downsampling - Fitting')
    classifier = classifier.fit(
        x_split[feature_cols].to_numpy(), y_split.to_numpy()
    )
    confidence = classifier.predict_proba(x_split[feature_cols].to_numpy())[
        :, VALIDATION_DATA
    ]
    auc = roc_auc_score(y_split, confidence)
    logger.info(f'Adversarial Downsampling - AUC: {auc:.2f}')
    results[COLUMN_DATA_TYPE_CONFIDENCE] = confidence

    # Select only training data
    training_results = results[
        results[COLUMN_DATA_TYPE_REFERENCE] == TRAINING_DATA
    ]

    # Grab rows with highest confidence as trainig
    downsampled_training_data = training_data.loc[
        training_results.groupby(COLUMN_ERA)
        .apply(
            lambda group: group[COLUMN_DATA_TYPE_CONFIDENCE]
            .nlargest(int(len(group) * downsampling_ratio))
            .index.tolist(),
            include_groups=False,
        )
        .explode()
        .tolist()
    ]
    return feature_cols, downsampled_training_data, validation_data
