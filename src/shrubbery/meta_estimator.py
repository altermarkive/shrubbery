from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin

from shrubbery.constants import COLUMN_INDEX_ERA, COLUMN_INDEX_TARGET
from shrubbery.data.augmentation import get_biggest_change_features
from shrubbery.neutralization import neutralize
from shrubbery.observability import logger
from shrubbery.utilities import PrintableModelMixin, load_model, store_model


# Numerai-specific estimator wrapper
# For a more elegant application of neutralization observe progress of:
# * https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html  # noqa: E501
# * https://scikit-learn.org/stable/developers/develop.html
# * https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep001/proposal.html  # noqa: E501
class NumeraiMetaEstimator(
    BaseEstimator, MetaEstimatorMixin, RegressorMixin, PrintableModelMixin
):
    def __init__(
        self,
        estimator: Any,
        drop_era_column: bool,
        target: int,
        neutralization_feature_indices: list[int] | None,
        neutralization_proportion: float,
        neutralization_normalize: bool,
        **kwargs: dict,
    ) -> None:
        self.estimator = estimator
        self.drop_era_column = drop_era_column
        self.target = target
        self.neutralization_feature_indices = neutralization_feature_indices
        self.neutralization_proportion = neutralization_proportion
        self.neutralization_normalize = neutralization_normalize

    def fit(
        self, x: np.ndarray, y: np.ndarray, **kwargs: dict[str, Any]
    ) -> 'NumeraiMetaEstimator':
        logger.info(f'Fitting {self}')
        logger.info(f'Shape of the data to train on: {x.shape} {y.shape}')
        feature_indices = list(range(COLUMN_INDEX_ERA + 1, x.shape[1]))
        self.estimator = self.estimator.fit(
            x[:, feature_indices] if self.drop_era_column else x,
            y[:, [self.target]].ravel(),
        )
        if self.neutralization_feature_indices is None:
            # Riskiest features
            riskiest_features = get_biggest_change_features(
                x,
                y[:, [COLUMN_INDEX_TARGET]],
                list(range(COLUMN_INDEX_ERA + 1, x.shape[1])),
                50,
            )
            self.neutralization_feature_indices = riskiest_features
            logger.info(f'Riskiest features (indices): {riskiest_features}')
        elif len(self.neutralization_feature_indices) == 0:
            self.neutralization_feature_indices = None
        self.fitted_ = True
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        logger.info(f'Predicting {self}')
        logger.info(f'Shape of the data to predict on: {x.shape}')
        feature_indices = list(range(COLUMN_INDEX_ERA + 1, x.shape[1]))
        predictions = self.estimator.predict(
            x[:, feature_indices] if self.drop_era_column else x
        )
        if self.neutralization_feature_indices is not None:
            neutralized = neutralize(
                x,
                predictions,
                self.neutralization_feature_indices,
                self.neutralization_proportion,
                self.neutralization_normalize,
            )
            neutralized = np.concatenate(
                [
                    x[:, COLUMN_INDEX_ERA].reshape(-1, 1),
                    neutralized.reshape(-1, 1),
                ],
                axis=1,
            )
            # Ranking per era for all of our columns so we can combine safely
            # on the same scales.
            # See also:
            # https://forum.numer.ai/t/neutralization-output-in-5-5-range/6324
            predictions_columns = [1]
            predictions = (
                pd.DataFrame(neutralized)
                .groupby(COLUMN_INDEX_ERA, group_keys=False)
                .apply(
                    lambda group: group[predictions_columns].rank(pct=True),
                    include_groups=False,
                )
                .to_numpy()
            )
        return predictions


class PersistentRegressor(
    BaseEstimator, MetaEstimatorMixin, RegressorMixin, PrintableModelMixin
):
    def __init__(
        self,
        estimator: Any,
        model_name: str,
        model_version: str = 'latest',
        **kwargs: dict,
    ) -> None:
        self.model_name = model_name
        self.model_version = model_version
        model, version = load_model(model_name, model_version)
        if model is not None:
            estimator = model
            self.model_version = version
        kwargs['estimator'] = estimator
        super().__init__(**kwargs)

    def fit(
        self, x: Any, y: Any, **kwargs: dict[str, Any]
    ) -> 'PersistentRegressor':
        super().fit(x, y)
        self.model_version = store_model(self.estimator, self.model_name)
        self.fitted_ = True
        return self

    def predict(self, x: Any) -> np.ndarray:
        predictions = self.estimator.predict(x)
        return predictions
