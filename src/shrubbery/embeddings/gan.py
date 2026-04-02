# Code inspired by:
# * https://github.com/jimfleming/numerai/blob/master/models/adversarial/model.py  # noqa: E501
# * https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/  # noqa: E501
# * https://medium.com/@mattiaspinelli/simple-generative-adversarial-network-gans-with-keras-1fe578e44a87  # noqa: E501
# * https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py  # noqa: E501
import io
import os

os.environ['KERAS_BACKEND'] = 'torch'

import keras.regularizers  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.jit as jit  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.init as init  # noqa: E402
from keras.initializers import VarianceScaling  # noqa: E402
from keras.layers import (  # noqa: E402
    Activation,
    BatchNormalization,
    Dense,
    Input,
    LeakyReLU,
)
from keras.models import Model, Sequential  # noqa: E402
from keras.ops import convert_to_tensor  # noqa: E402
from keras.optimizers import Adam  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.utils import shuffle  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

from shrubbery.utilities import (  # noqa: E402
    deserialize_keras_model,
    serialize_keras_model,
)


def create_discriminator(feature_count: int, layer_units: list[int]) -> Model:
    softmax_init = VarianceScaling(
        scale=1.0, mode='fan_in', distribution='truncated_normal'
    )
    weights_reg = keras.regularizers.l2(1e-3)
    discriminator: Model = Sequential(name='discriminator')
    discriminator.add(Input(shape=(feature_count,)))
    all_layer_units = layer_units + [1]  # Adding logits to embedder
    for i, units in enumerate(all_layer_units):
        discriminator.add(
            Dense(
                units,
                kernel_initializer=softmax_init,
                kernel_regularizer=weights_reg,
            )
        )
        # Placing normalization before activation may:
        # * stabilize training
        # * improve activation performance (works better normalized inputs)
        # * convergence faster and get better results
        discriminator.add(BatchNormalization())
        if i < len(all_layer_units) - 1:
            # Using ReLU (instead of sigmoid) on hidden layers may help
            # with faster and more efficient training. LeakyReLU addresses
            # the issue of "dying ReLUs" and may help maintaining non-zero
            # gradients and improve learning dynamics.
            discriminator.add(LeakyReLU(negative_slope=0.2))
    discriminator.summary()
    return discriminator


def create_generator(
    latent_dim: int, layer_units: list[int], feature_count: int
) -> Model:
    sigmoid_init = VarianceScaling(
        scale=1.0, mode='fan_in', distribution='truncated_normal'
    )
    weights_reg = keras.regularizers.l2(1e-3)
    generator: Model = Sequential(name='generator')
    generator.add(Input(shape=(latent_dim,)))
    all_layer_units = layer_units + [feature_count]
    for i, units in enumerate(all_layer_units):
        generator.add(
            Dense(
                units,
                kernel_initializer=sigmoid_init,
                kernel_regularizer=weights_reg,
            )
        )
        # Placing normalization before activation may:
        # * stabilize training
        # * improve activation performance (works better normalized inputs)
        # * convergence faster and get better results
        generator.add(BatchNormalization())
        # Using ReLU (instead of sigmoid) on hidden layers may help
        # with faster and more efficient training. LeakyReLU addresses
        # the issue of "dying ReLUs" and may help maintaining non-zero
        # gradients and improve learning dynamics.
        generator.add(
            LeakyReLU(negative_slope=0.2)
            if i < len(all_layer_units) - 1
            else Activation('sigmoid')
        )
    generator.summary()
    return generator


class GANEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        latent_dim: int,
        generator_layer_units: list[int],
        discriminator_layer_units: list[int],
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.generator_layer_units = generator_layer_units
        self.discriminator_layer_units = discriminator_layer_units
        self.learning_rate = 1e-4

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'GANEmbedder':
        # GAN
        feature_count = x.shape[1]
        discriminator: Model = create_discriminator(
            feature_count, self.discriminator_layer_units
        )
        discriminator.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy'],
        )
        generator: Model = create_generator(
            self.latent_dim,
            self.generator_layer_units,
            feature_count,
        )
        discriminator.trainable = False
        model: Model = Sequential(name='gan')
        model.add(generator)
        model.add(discriminator)
        model.summary()
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
        )
        # Training
        for _ in range(self.epochs):
            x_epoch, y_epoch = shuffle(x, y)
            for i in (
                progress := tqdm(range(0, len(x_epoch), self.batch_size))
            ):
                # Train discriminator
                discriminator.trainable = True
                x_batch = x_epoch[i : i + self.batch_size]
                # y_batch = y_epoch[i : i + self.batch_size]
                g_noise = np.random.normal(
                    size=(x_batch.shape[0], self.latent_dim),
                    loc=0.0,
                    scale=1.0,
                )
                syntetic_features = generator.predict(
                    convert_to_tensor(g_noise), verbose=0
                )
                x_combined = np.concatenate((x_batch, syntetic_features))
                y_combined = np.concatenate(
                    (
                        np.ones((x_batch.shape[0], 1)),
                        np.zeros((x_batch.shape[0], 1)),
                    )
                )
                d_loss = discriminator.train_on_batch(
                    convert_to_tensor(x_combined),
                    convert_to_tensor(y_combined),
                )[
                    0  # Keep only loss
                ]
                # Train generator
                discriminator.trainable = False
                d_noise = np.random.normal(
                    size=(2 * x_batch.shape[0], self.latent_dim),
                    loc=0.0,
                    scale=1.0,
                )
                y_mislabled = np.ones((2 * x_batch.shape[0], 1))
                g_loss = model.train_on_batch(
                    convert_to_tensor(d_noise), convert_to_tensor(y_mislabled)
                )
                progress.set_description(
                    f'Training - d_loss: {d_loss:.5f}; g_loss: {g_loss:.5f}'
                )
        embedder = Sequential(
            discriminator.layers[:-3]
        )  # Removed logit layers from discriminator
        embedder.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
        )
        self.serialized_embedder_ = serialize_keras_model(embedder)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.serialized_embedder_ is not None
        embedder = deserialize_keras_model(
            self.serialized_embedder_,
            # custom_objects={
            #     'create_discriminator': create_discriminator,
            #     'create_generator': create_generator,
            # },
        )
        result = embedder.predict(convert_to_tensor(x))
        return result


def variance_scaling_initializer_with_fan_in(module: nn.Module) -> None:
    """Initialize weights using variance scaling (with fan-in and factor of 1.0)."""
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            fan_in = submodule.weight.size(1)
            std = (1.0 / fan_in) ** 0.5
            init.trunc_normal_(
                submodule.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
            )
            if submodule.bias is not None:
                init.zeros_(submodule.bias)


class DiscriminatorNetwork(nn.Module):
    def __init__(self, feature_count: int, layer_units: list[int]) -> None:
        super().__init__()
        all_layer_units = layer_units + [1]  # Adding logits layer
        discriminator_layers: list[nn.Module] = []
        previous_units = feature_count
        for i, units in enumerate(all_layer_units):
            discriminator_layers.append(nn.Linear(previous_units, units))
            # Placing normalization before activation may:
            # * stabilize training
            # * improve activation performance (works better normalized inputs)
            # * convergence faster and get better results
            discriminator_layers.append(nn.BatchNorm1d(units))
            if i < len(all_layer_units) - 1:
                # Using ReLU (instead of sigmoid) on hidden layers may help
                # with faster and more efficient training. LeakyReLU addresses
                # the issue of "dying ReLUs" and may help maintaining non-zero
                # gradients and improve learning dynamics.
                discriminator_layers.append(nn.LeakyReLU(negative_slope=0.2))
            previous_units = units
        self.discriminator = nn.Sequential(*discriminator_layers)
        variance_scaling_initializer_with_fan_in(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.discriminator(x)


class GeneratorNetwork(nn.Module):
    def __init__(
        self, latent_dim: int, layer_units: list[int], feature_count: int
    ) -> None:
        super().__init__()
        all_layer_units = layer_units + [feature_count]
        generator_layers: list[nn.Module] = []
        previous_units = latent_dim
        for i, units in enumerate(all_layer_units):
            generator_layers.append(nn.Linear(previous_units, units))
            # Placing normalization before activation may:
            # * stabilize training
            # * improve activation performance (works better normalized inputs)
            # * convergence faster and get better results
            generator_layers.append(nn.BatchNorm1d(units))
            # Using ReLU (instead of sigmoid) on hidden layers may help
            # with faster and more efficient training. LeakyReLU addresses
            # the issue of "dying ReLUs" and may help maintaining non-zero
            # gradients and improve learning dynamics.
            generator_layers.append(
                nn.LeakyReLU(negative_slope=0.2)
                if i < len(all_layer_units) - 1
                else nn.Sigmoid()
            )
            previous_units = units
        self.generator = nn.Sequential(*generator_layers)
        variance_scaling_initializer_with_fan_in(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.generator(x)


class GenerativeAdversarialNetworkEmbedder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        latent_dim: int,
        generator_layer_units: list[int],
        discriminator_layer_units: list[int],
        learning_rate: float,
        device: str,
    ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.latent_dim = latent_dim
        self.generator_layer_units = generator_layer_units
        self.discriminator_layer_units = discriminator_layer_units
        self.learning_rate = learning_rate
        self.device = device

    def fit(
        self, x: np.ndarray, y: np.ndarray
    ) -> 'GenerativeAdversarialNetworkEmbedder':
        # GAN
        feature_count = x.shape[1]
        discriminator = DiscriminatorNetwork(
            feature_count, self.discriminator_layer_units
        ).to(self.device)
        d_optimizer = torch.optim.Adam(
            discriminator.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-3,
        )
        generator = GeneratorNetwork(
            self.latent_dim,
            self.generator_layer_units,
            feature_count,
        ).to(self.device)
        g_optimizer = torch.optim.Adam(
            generator.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-3,
        )
        criterion = nn.BCEWithLogitsLoss()
        # Training
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in (progress := tqdm(range(self.epochs))):
            discriminator.train()
            generator.train()
            for x_batch, y_batch in loader:
                batch_size = x_batch.size(0)
                # Train discriminator
                d_optimizer.zero_grad()
                real_labels = torch.ones(batch_size, 1).to(self.device)
                real_outputs = discriminator(x_batch)
                d_loss_real = criterion(real_outputs, real_labels)
                g_noise = torch.randn(batch_size, self.latent_dim).to(
                    self.device
                )
                syntetic_features = generator(g_noise)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                fake_outputs = discriminator(syntetic_features.detach())
                d_loss_fake = criterion(fake_outputs, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                # Train generator
                g_optimizer.zero_grad()
                d_noise = torch.randn(2 * batch_size, self.latent_dim).to(
                    self.device
                )
                fake_samples = generator(d_noise)
                fake_outputs = discriminator(fake_samples)
                y_mislabled = torch.ones(2 * batch_size, 1).to(self.device)
                g_loss = criterion(fake_outputs, y_mislabled)
                g_loss.backward()
                g_optimizer.step()
            progress.set_description(
                f'Training - epoch: {epoch}; '
                f'd_loss: {d_loss.item():.5f}; g_loss: {g_loss.item():.5f}'
            )
        # Extract embedder from discriminator (remove last 2 layers, earlier ones have 3)
        # Removes: final Linear & BatchNorm
        # Keeps: all layers up to and including the last hidden LeakyReLU
        embedder_layers = list(discriminator.discriminator.children())[:-2]
        embedder = nn.Sequential(*embedder_layers)
        # Serialize the embedder
        self.serialized_model_ = io.BytesIO()
        jit.save(jit.script(embedder), self.serialized_model_)
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
