from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from shrubbery.adapter import TorchRegressor


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float) -> None:
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dense1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout1(out)
        out = self.dense2(out)
        out = self.bn2(out)
        out = self.dropout2(out)
        out += residual  # Skip connection
        out = self.activation(out)
        return out


class ResNetRegressor(TorchRegressor):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        hidden_dim: int,
        num_blocks: int,
        dropout_rate: float,
        learning_rate: float,
        weight_decay: float,
        device: str,
    ) -> None:
        super().__init__(epochs=epochs, device=device)
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
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
        layers: list[nn.Module] = []

        # Input projection to hidden dimension
        layers.append(nn.Linear(input_dim, self.hidden_dim))
        layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        # Residual blocks
        for _ in range(self.num_blocks):
            layers.append(ResidualBlock(self.hidden_dim, self.dropout_rate))

        # Output projection
        output_dense = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(output_dense.bias)
        layers.append(output_dense)
        layers.append(nn.Sigmoid())

        module = nn.Sequential(*layers)

        # Optimizer & criterion
        optimizer = optim.AdamW(
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()
        return (module, optimizer, criterion)
