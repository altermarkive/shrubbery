from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset


def relu_initializer(tensor):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    scale = 2.0 / max(1.0, fan_in)
    std = scale**0.5
    return nn.init.normal_(tensor, mean=0.0, std=std)


def output_initializer(tensor):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    scale = 1.0 / max(1.0, fan_in)
    std = scale**0.5
    return nn.init.normal_(tensor, mean=0.0, std=std)


class TorchRegressor(BaseEstimator, RegressorMixin, ABC):
    def __init__(
        self,
        epochs: int,
        device: str,
    ) -> None:
        self.epochs = epochs
        self.device = device

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'TorchRegressor':
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        module, optimizer, criterion = self.prepare(input_dim=x.shape[1])
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for _ in range(self.epochs):
            module.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = module(x_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                loss.backward()
                optimizer.step()
        self.serialized_model_ = jit.script(module)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.serialized_model_.eval()
        with torch.no_grad():
            predictions = (
                self.serialized_model_(x_tensor).cpu().numpy().squeeze()
            )
        return predictions

    @abstractmethod
    def prepare(
        self, input_dim: int
    ) -> tuple[
        nn.Module,
        torch.optim.Optimizer,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        pass


class FeedforwardNeuralNetworkModule(TorchRegressor):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        layer_units: list[int],
        device: str,
    ) -> None:
        super().__init__(epochs=epochs, device=device)
        self.batch_size = batch_size
        self.layer_units = layer_units
        self.learning_rate = 1e-4

    def prepare(
        self, input_dim: int
    ) -> tuple[
        nn.Module,
        torch.optim.Optimizer,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        layers: list[nn.Module] = []
        # Embedder / feature extractor
        for units in self.layer_units:
            dense = nn.Linear(input_dim, units)
            relu_initializer(dense.weight)
            layers.append(dense)
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.LeakyReLU())
            input_dim = units
        # Regressor
        dense = nn.Linear(input_dim, 1)
        output_initializer(dense.weight)
        layers.append(dense)
        layers.append(nn.BatchNorm1d(1))
        module = nn.Sequential(*layers)
        # Optimizer & criterion
        optimizer = optim.Adam(
            module.parameters(), lr=self.learning_rate, weight_decay=1e-3
        )
        criterion = nn.MSELoss()
        return (module, optimizer, criterion)
