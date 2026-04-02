from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from shrubbery.adapter import TorchRegressor


def relu_initializer(tensor: torch.Tensor) -> torch.Tensor:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    scale = 2.0 / max(1.0, fan_in)
    std = scale**0.5
    return nn.init.normal_(tensor, mean=0.0, std=std)


def output_initializer(tensor: torch.Tensor) -> torch.Tensor:
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    scale = 1.0 / max(1.0, fan_in)
    std = scale**0.5
    return nn.init.normal_(tensor, mean=0.0, std=std)


def mse_with_weight_regularization(
    module: nn.Module, scale: float
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    mse = nn.MSELoss()

    def l2_regularization(
        y_prediction: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        reg_loss = torch.tensor(0.0)
        for param in module.parameters():
            if param.ndim > 1:
                reg_loss += torch.sum(param**2)
        return mse(y_prediction, y_true) + scale * reg_loss

    return l2_regularization


class FeedforwardNeuralNetworkRegressor(TorchRegressor):
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
        nn.init.zeros_(dense.bias)
        layers.append(dense)
        module = nn.Sequential(*layers)
        # Optimizer & criterion
        optimizer = optim.Adam(
            module.parameters(), lr=self.learning_rate, weight_decay=0.0
        )
        criterion = mse_with_weight_regularization(module, 1e-3)
        return (module, optimizer, criterion)
