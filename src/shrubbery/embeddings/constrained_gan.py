# Code inspired by:
# * https://github.com/jimfleming/numerai/blob/master/models/adversarial/model.py  # noqa: E501
# * https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-a-1-dimensional-function-from-scratch-in-keras/  # noqa: E501
# * https://medium.com/@mattiaspinelli/simple-generative-adversarial-network-gans-with-keras-1fe578e44a87  # noqa: E501
# * https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py  # noqa: E501
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from shrubbery.adapter import (
    CompilerBackend,
    ModelWrapper,
    TorchEstimator,
    variance_scaling_initializer_with_fan_in,
)

# Why BatchNorm is applied to every layer (including the final one) of both
# the discriminator and the generator:
#
# * Discriminator final layer (Linear + BatchNorm, no activation):
#   BCEWithLogitsLoss consumes raw logits. Applying BatchNorm to those logits
#   normalizes them to roughly N(0, 1), which caps how confident the
#   discriminator can become and prevents logit saturation. This acts as
#   implicit output regularization, similar in spirit to (a much lighter
#   form of) spectral normalization or gradient penalty.
#
# * Why this matters for embedding quality:
#   In standard GAN training, an unconstrained discriminator tends to overfit
#   by finding trivial shortcuts to separate real from fake, collapsing onto
#   a shallow decision boundary. The BatchNorm constraint forces it to stay
#   in a moderate-confidence regime, so to maintain discrimination ability
#   it must build richer, more informative hidden representations. Those
#   hidden layers are exactly what is extracted as the embedding, so a
#   constrained discriminator yields higher-quality embeddings.
#
# * Generator final layer (Linear + BatchNorm + Sigmoid):
#   BatchNorm before Sigmoid produces inputs near N(0, 1), so the Sigmoid
#   maps them to a spread around 0.5 instead of saturating at 0 or 1. The
#   resulting synthetic samples look more in-distribution, making the
#   discriminator's task genuinely harder and reinforcing the pressure on
#   the discriminator to learn better features.
#
# * Discriminator learning rate is NOT halved relative to the generator:
#   The BatchNorm-based output constraint already prevents the discriminator
#   from dominating, so an additional lr reduction would only slow learning
#   of the hidden representations and degrade embedding quality.


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
            # * improve activation performance (better normalized inputs)
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


class GenerativeAdversarialNetworkEmbedder(TorchEstimator):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        latent_dim: int,
        generator_layer_units: list[int],
        discriminator_layer_units: list[int],
        learning_rate: float,
        device: str,
        compiler: CompilerBackend,
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            compiler=compiler,
        )
        self.latent_dim = latent_dim
        self.generator_layer_units = generator_layer_units
        self.discriminator_layer_units = discriminator_layer_units
        self.learning_rate = learning_rate

    def train(self, x: torch.Tensor, y: torch.Tensor) -> nn.Module:
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
        d_scaler = GradScaler(self.device)
        g_scaler = GradScaler(self.device)
        # Training
        dataset = TensorDataset(x)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in (progress := tqdm(range(self.epochs))):
            discriminator.train()
            generator.train()
            for (x_batch,) in loader:
                batch_size = x_batch.size(0)
                # Train discriminator
                discriminator.train()
                d_optimizer.zero_grad()
                with autocast(device_type=self.device, dtype=torch.float16):
                    g_noise = torch.randn(batch_size, self.latent_dim).to(
                        self.device
                    )
                    synthetic_features = generator(g_noise)
                    x_combined = torch.cat(
                        [x_batch, synthetic_features.detach()], dim=0
                    )
                    y_combined = torch.cat(
                        [
                            torch.ones(batch_size, 1),
                            torch.zeros(batch_size, 1),
                        ],
                        dim=0,
                    ).to(self.device)
                    d_outputs = discriminator(x_combined)
                    d_loss = criterion(d_outputs, y_combined)
                d_scaler.scale(d_loss).backward()
                d_scaler.step(d_optimizer)
                d_scaler.update()
                # Train generator
                discriminator.eval()
                g_optimizer.zero_grad()
                with autocast(device_type=self.device, dtype=torch.float16):
                    d_noise = torch.randn(2 * batch_size, self.latent_dim).to(
                        self.device
                    )
                    fake_samples = generator(d_noise)
                    fake_outputs = discriminator(fake_samples)
                    y_mislabeled = torch.ones(2 * batch_size, 1).to(
                        self.device
                    )
                    g_loss = criterion(fake_outputs, y_mislabeled)
                g_scaler.scale(g_loss).backward()
                g_scaler.step(g_optimizer)
                g_scaler.update()
            progress.set_description(
                f'Training - epoch: {epoch}; '
                f'd_loss: {d_loss.item():.5f}; g_loss: {g_loss.item():.5f}'
            )
        # Extract embedder from discriminator (remove last 2 layers, earlier ones have 3)
        # Removes: final Linear & BatchNorm
        # Keeps: all layers up to and including the last hidden LeakyReLU
        embedder_layers = list(discriminator.discriminator.children())[:-2]
        embedder = nn.Sequential(*embedder_layers)
        return ModelWrapper(embedder)

    def module(self, input_dim: int) -> nn.Module:
        discriminator = DiscriminatorNetwork(
            input_dim, self.discriminator_layer_units
        )
        embedder_layers = list(discriminator.discriminator.children())[:-2]
        return nn.Sequential(*embedder_layers)
