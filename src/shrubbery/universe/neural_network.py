# Code inspired by: https://github.com/jimfleming/numerai/blob/master/models/classifier/model.py  # noqa: E501
import os

os.environ['KERAS_BACKEND'] = 'torch'

import keras.ops  # noqa: E402
import keras.regularizers  # noqa: E402
import numpy as np  # noqa: E402
from keras.initializers import VarianceScaling  # noqa: E402
from keras.layers import (  # noqa: E402
    BatchNormalization,
    Dense,
    Input,
    LeakyReLU,
)
from keras.models import Model, Sequential  # noqa: E402
from keras.ops import convert_to_tensor  # noqa: E402
from keras.optimizers import Adam  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402

from shrubbery.utilities import (  # noqa: E402
    deserialize_keras_model,
    serialize_keras_model,
)


class NeuralNetwork(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        layer_units: list[int],
    ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.layer_units = layer_units
        self.learning_rate = 1e-4

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'NeuralNetwork':
        model_columns = list(range(x.shape[1]))

        optimizer = Adam(learning_rate=self.learning_rate)

        model: Model = Sequential()
        model_input = Input(shape=(len(model_columns),))
        model.add(model_input)
        # Embedder / feature extractor
        relu_init = 'glorot_normal'
        for units in self.layer_units:
            model.add(
                Dense(
                    units,
                    kernel_initializer=relu_init,
                    kernel_regularizer=keras.regularizers.l2(1e-3),
                )
            )
            model.add(BatchNormalization())
            model.add(LeakyReLU())
        # Regressor
        output_init = VarianceScaling(
            scale=1.0, mode='fan_in', distribution='truncated_normal'
        )
        model.add(
            Dense(
                1,
                kernel_initializer=output_init,
                kernel_regularizer=keras.regularizers.l2(1e-3),
            )
        )
        model.add(BatchNormalization())
        # Training
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        model.fit(
            convert_to_tensor(x),
            convert_to_tensor(y),
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        self.serialized_model_ = serialize_keras_model(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.serialized_model_ is not None
        model = deserialize_keras_model(self.serialized_model_)
        result = model.predict(convert_to_tensor(x))
        return result
