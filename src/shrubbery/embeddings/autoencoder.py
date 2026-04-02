# Code inspired by: https://github.com/jimfleming/numerai/blob/master/models/autoencoder/model.py  # noqa: E501
import io
import os

os.environ['KERAS_BACKEND'] = 'torch'

import keras.ops  # noqa: E402
import keras.random  # noqa: E402
import keras.regularizers  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.jit as jit  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.init as init  # noqa: E402
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
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from tqdm import tqdm  # noqa: E402


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
        embedder.build(input_shape=(None, x.shape[1]))
        embedder_config = embedder.get_config()
        embedder_weights = embedder.get_weights()
        self.serialized_embedder_ = (embedder_config, embedder_weights)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.serialized_embedder_ is not None
        embedder_config, embedder_weights = self.serialized_embedder_
        embedder = Sequential.from_config(embedder_config)
        embedder.set_weights(embedder_weights)
        result = embedder.predict(convert_to_tensor(x))
        return result


class AutoencoderNetwork(nn.Module):
    def __init__(self, input_dim: int, layer_units: list[int]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer_units = layer_units
        # Encoder
        encoder_layer_units = layer_units
        encoder_layers = []
        previous_units = input_dim
        for units in encoder_layer_units:
            encoder_layers.extend(
                [
                    nn.Linear(previous_units, units),
                    # TODO: Check if 1e-3 would have better performance
                    nn.BatchNorm1d(units, eps=1e-5),
                    nn.Sigmoid(),
                ]
            )
            previous_units = units
        # Decoder (symmetric)
        decoder_layer_units = layer_units[:-1][::-1] + [input_dim]
        decoder_layers = []
        for units in decoder_layer_units:
            decoder_layers.extend(
                [
                    nn.Linear(previous_units, units),
                    # TODO: Check if 1e-3 would have better performance
                    nn.BatchNorm1d(units, eps=1e-5),
                    nn.Sigmoid(),
                ]
            )
            prev_units = units
        # Two parts of the model
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.size(1)
                std = (1.0 / fan_in) ** 0.5
                init.trunc_normal_(
                    module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
                )
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        layer_units: list[int],
        denoise: bool,
        learning_rate: float,
        device: str,
    ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.layer_units = layer_units
        self.denoise = denoise
        self.learning_rate = learning_rate
        self.device = device

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'AutoencoderEmbedder':
        # Autoencoder
        input_dim = x.shape[1]
        module = AutoencoderNetwork(input_dim, self.layer_units).to(
            self.device
        )
        # Training
        x_train = x.copy()
        if self.denoise:
            x_variance = np.var(x, axis=0)
            x_stddev = np.sqrt(x_variance)
            noise = np.random.normal(0, 0.1 * x_stddev, size=x.shape)
            x_train = x_train + noise
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-3,
        )
        criterion = nn.MSELoss()
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in (progress := tqdm(range(self.epochs))):
            module.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = module(x_batch)
                metric = criterion(outputs, y_batch)
                metric.backward()
                torch.nn.utils.clip_grad_norm_(
                    module.parameters(), max_norm=1.0
                )
                optimizer.step()
            progress.set_description(
                f'Training - epoch: {epoch}; metric: {metric:.5f}'
            )
        self.serialized_model_ = io.BytesIO()
        jit.save(jit.script(module.encoder), self.serialized_model_)
        self.serialized_model_.seek(0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.serialized_model_.seek(0)
        model = torch.jit.load(self.serialized_model_)
        self.serialized_model_.seek(0)
        model.eval()
        with torch.no_grad():
            result = model(x_tensor).cpu().numpy().squeeze()
        return result
