from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from shrubbery.adapter import TorchRegressor


def mse_with_weight_regularization(
    module: nn.Module, scale: float, device: str
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    mse = nn.MSELoss()

    def l2_regularization(
        y_prediction: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        reg_loss = torch.tensor(0.0).to(device)
        for param in module.parameters():
            if param.ndim > 1:
                reg_loss += torch.sum(param**2)
        return mse(y_prediction, y_true) + scale * reg_loss

    return l2_regularization


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dense1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dense2(out)
        out = self.bn2(out)
        out = out + residual  # Skip connection
        out = self.activation(out)
        return out


class ResNetRegressor(TorchRegressor):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        hidden_dim: int,
        num_blocks: int,
        device: str,
    ) -> None:
        super().__init__(epochs=epochs, device=device)
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.learning_rate = 1e-4

    def prepare(
        self, input_dim: int
    ) -> tuple[
        nn.Module,
        torch.optim.Optimizer,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        layers: list[nn.Module] = []

        # Input projection to hidden dimension
        input_dense = nn.Linear(input_dim, self.hidden_dim)
        layers.append(input_dense)
        layers.append(nn.BatchNorm1d(self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))

        # Residual blocks
        for _ in range(self.num_blocks):
            layers.append(ResidualBlock(self.hidden_dim))

        # Output projection
        output_dense = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(output_dense.bias)
        layers.append(output_dense)
        layers.append(nn.Sigmoid())

        module = nn.Sequential(*layers)

        # Optimizer & criterion
        optimizer = optim.Adam(
            module.parameters(), lr=self.learning_rate, weight_decay=0.0
        )
        criterion = mse_with_weight_regularization(
            module, 1e-3, self.device
        )
        return (module, optimizer, criterion)
