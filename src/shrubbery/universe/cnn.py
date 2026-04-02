from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from shrubbery.adapter import TorchRegressor


class CNNModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_filters: int,
        kernel_size: int,
        dense_units: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm1d(num_filters)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(num_filters * input_dim, dense_units)
        self.dense_bn = nn.BatchNorm1d(dense_units)
        self.dense_relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(dense_units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.conv1d(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dense_bn(x)
        x = self.dense_relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class CNNRegressor(TorchRegressor):
    def __init__(
        self,
        num_filters: int,
        kernel_size: int,
        dense_units: int,
        dropout_rate: float,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        device: str,
    ) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, device=device)
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def prepare(
        self, input_dim: int
    ) -> tuple[
        nn.Module,
        torch.optim.Optimizer,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        module = CNNModule(
            input_dim=input_dim,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            dense_units=self.dense_units,
            dropout_rate=self.dropout_rate,
        )
        optimizer = optim.AdamW(
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.MSELoss()
        return (module, optimizer, criterion)
