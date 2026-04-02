# Code inspired by: https://github.com/jimfleming/numerai/blob/master/models/autoencoder/model.py  # noqa: E501
import io

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from shrubbery.adapter import (
    variance_scaling_initializer_with_fan_in,
)


class AutoencoderNetwork(nn.Module):
    def __init__(
        self, input_dim: int, layer_units: list[int], batch_norm_eps: float
    ) -> None:
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
                    nn.BatchNorm1d(units, eps=batch_norm_eps),
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
                    nn.BatchNorm1d(units, eps=batch_norm_eps),
                    nn.Sigmoid(),
                ]
            )
            previous_units = units
        # Two parts of the model
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        # Initialize weights
        variance_scaling_initializer_with_fan_in(self)

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
        batch_norm_eps: float,
        device: str,
    ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.layer_units = layer_units
        self.denoise = denoise
        self.learning_rate = learning_rate
        self.batch_norm_eps = batch_norm_eps
        self.device = device

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'AutoencoderEmbedder':
        # Autoencoder
        input_dim = x.shape[1]
        module = AutoencoderNetwork(
            input_dim, self.layer_units, self.batch_norm_eps
        ).to(self.device)
        # Training
        if self.denoise:
            x_variance = np.var(x, axis=0)
            x_stddev = np.sqrt(x_variance)
        y_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-3,
        )
        criterion = nn.MSELoss()
        for epoch in range(self.epochs):
            if self.denoise:
                noise = np.random.normal(0, 0.1 * x_stddev, size=x.shape)
                x_train = x + noise
            else:
                x_train = x
            x_tensor = torch.tensor(x_train, dtype=torch.float32).to(
                self.device
            )
            dataset = TensorDataset(x_tensor, y_tensor)
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )
            module.train()
            metric_sum = 0.0
            for i, (x_batch, y_batch) in enumerate(progress := tqdm(loader)):
                optimizer.zero_grad()
                outputs = module(x_batch)
                metric = criterion(outputs, y_batch)
                metric.backward()
                torch.nn.utils.clip_grad_norm_(
                    module.parameters(), max_norm=1.0
                )
                optimizer.step()
                metric_sum += metric.item()
                metric_average = metric_sum / (i + 1)
                progress.set_description(
                    f'Training - epoch: {epoch}; loss: {metric_average:.4f}'
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
