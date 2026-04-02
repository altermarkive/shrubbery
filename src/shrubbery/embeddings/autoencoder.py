# Code inspired by: https://github.com/jimfleming/numerai/blob/master/models/autoencoder/model.py  # noqa: E501
import os

os.environ['KERAS_BACKEND'] = 'torch'

import keras.ops  # noqa: E402
import keras.random  # noqa: E402
import keras.regularizers  # noqa: E402
from keras.initializers import VarianceScaling  # noqa: E402
from keras.layers import (
    Activation,
    BatchNormalization,
    Dense,
    Input,
)  # noqa: E402
from keras.losses import MeanSquaredError  # noqa: E402
from keras.models import Model, Sequential  # noqa: E402
from keras.ops import convert_to_tensor  # noqa: E402
from keras.optimizers import Adam  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402

from shrubbery.utilities import (  # noqa: E402
    deserialize_keras_model,
    serialize_keras_model,
)


class Autoencoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        layer_units: list[int],
        denoise: bool,
        learning_rate: float,
    ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.layer_units = layer_units
        self.denoise = denoise
        self.learning_rate = learning_rate

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'Autoencoder':
        model: Model = Sequential()
        model_input = Input(shape=(x.shape[1],))
        model.add(model_input)
        # Autoencoder
        sigmoid_init = VarianceScaling(
            scale=1.0, mode='fan_in', distribution='truncated_normal'
        )
        all_layer_units = self.layer_units + self.layer_units[:-1][::-1]
        all_layer_units = all_layer_units + [x.shape[1]]  # Reconstruction
        for units in all_layer_units:
            model.add(
                Dense(
                    units,
                    kernel_initializer=sigmoid_init,
                    kernel_regularizer=keras.regularizers.l2(1e-3),
                )
            )
            model.add(BatchNormalization(epsilon=1e-5))
            model.add(Activation('sigmoid'))
        # Training
        if self.denoise:
            _, x_variance = keras.ops.moments(x, axes=[0])
            x_stddev = keras.ops.sqrt(x_variance).tolist()
            x_shape = keras.ops.shape(x)
            for i, x_stddev_i in enumerate(x_stddev):
                x[:, i] += (
                    keras.random.normal((x_shape[0],), stddev=x_stddev_i * 0.1)
                    .cpu()
                    .numpy()
                    .ravel()
                )
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss=MeanSquaredError())
        model.fit(
            convert_to_tensor(x),
            convert_to_tensor(x),
            batch_size=self.batch_size,
            epochs=self.epochs,
        )
        embedder = Sequential(model.layers[: 2 * len(self.layer_units)])
        embedder.compile(optimizer=optimizer, loss=MeanSquaredError())
        self.serialized_embedder_ = serialize_keras_model(embedder)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.serialized_embedder_ is not None
        embedder = deserialize_keras_model(self.serialized_embedder_)
        result = embedder.predict(convert_to_tensor(x))
        return result
