# Code inspired by inspired by: https://github.com/Jeremy123W/Numerai
# Improved with help of: https://keras.io/examples/structured_data/wide_deep_cross_networks/  # noqa: E501
# As an alternative to the from-scratch approach one can also use:
# * https://github.com/jrzaurin/pytorch-widedeep
# Wide and Deep Learning research paper: https://arxiv.org/abs/1606.07792
# Wide & Deep Learning for RecSys with Pytorch: https://www.kaggle.com/code/matanivanov/wide-deep-learning-for-recsys-with-pytorch  # noqa: E501
from enum import Enum
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from ..adapter import CompilerBackend, TorchEstimator


class ModelType(str, Enum):
    WIDE = 'wide'
    DEEP = 'deep'
    WIDE_AND_DEEP = 'wide_and_deep'


class OptimizerType(str, Enum):
    SGD = 'sgd'
    ADAM = 'adam'
    ADAGRAD = 'adagrad'


class WideModule(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        # Build a wide model using Sequential API
        # (linear regression with no activation)
        layers: list[nn.Module] = []
        layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DeepModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: list[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        # Build a deep model using Sequential API
        # (focusing solely on a deep neural network for regression)
        layers: list[nn.Module] = []
        for unit in units:
            layers.append(nn.Linear(input_dim, unit))
            layers.append(nn.BatchNorm1d(unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = unit
        # Output layer, no activation for regression
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class WideAndDeepModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: list[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        # The Wide and Deep architecture is a specific neural network
        # architecture introduced by Google for handling a combination
        # of memorization
        # (shallow, wide paths for memorizing sparse features)
        # and generalization
        # (deep paths for learning abstract representations).
        #
        # Define the deep path
        deep_layers: list[nn.Module] = []
        deep_input_dim = input_dim
        for unit in units:
            deep_layers.append(nn.Linear(deep_input_dim, unit))
            deep_layers.append(nn.BatchNorm1d(unit))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            deep_input_dim = unit
        self.deep_model = nn.Sequential(*deep_layers)
        # Define the wide path (linear model)
        wide_layers: list[nn.Module] = []
        wide_input_dim = input_dim
        wide_layers.append(nn.BatchNorm1d(wide_input_dim))
        self.wide_model = nn.Sequential(*wide_layers)
        # Final output layer
        self.output_layer = nn.Linear(deep_input_dim + wide_input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_output = self.wide_model(x)
        deep_output = self.deep_model(x)
        # Concatenate deep and wide paths
        combined = torch.cat([wide_output, deep_output], dim=1)
        return self.output_layer(combined)


def mse_with_l1_regularization(
    module: nn.Module, l1_scale: float
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    mse = nn.MSELoss()

    def regularized_loss(
        y_prediction: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        l1_loss = sum(param.abs().sum() for param in module.parameters())
        return mse(y_prediction, y_true) + l1_scale * l1_loss

    return regularized_loss


class WideAndDeepRegressor(TorchEstimator):
    def __init__(
        self,
        model_type: ModelType,
        batch_size: int,
        epochs: int,
        dropout_rate: float,
        units: list[int],
        optimizer_type: OptimizerType,
        optimizer_learning_rate: float,
        optimizer_l1_regularization_strength: float,
        optimizer_l2_regularization_strength: float,
        device: str,
        compiler: CompilerBackend,
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            compiler=compiler,
        )
        self.model_type = model_type
        self.dropout_rate = dropout_rate
        self.units = units
        self.optimizer_type = optimizer_type
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer_l1_regularization_strength = (
            optimizer_l1_regularization_strength
        )
        self.optimizer_l2_regularization_strength = (
            optimizer_l2_regularization_strength
        )

    def module(self, input_dim: int) -> nn.Module:
        module: nn.Module
        match self.model_type:
            case ModelType.WIDE:
                module = WideModule(input_dim)
            case ModelType.DEEP:
                module = DeepModule(
                    input_dim,
                    self.units,
                    self.dropout_rate,
                )
            case ModelType.WIDE_AND_DEEP:
                module = WideAndDeepModule(
                    input_dim,
                    self.units,
                    self.dropout_rate,
                )
            case _:
                raise NotImplementedError()
        return module

    def prepare(
        self, model: nn.Module
    ) -> tuple[
        torch.optim.Optimizer,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        optimizer: torch.optim.Optimizer
        match self.optimizer_type:
            case OptimizerType.SGD:
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=self.optimizer_learning_rate,
                    weight_decay=self.optimizer_l2_regularization_strength,
                )
            case OptimizerType.ADAM:
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=self.optimizer_learning_rate,
                    weight_decay=self.optimizer_l2_regularization_strength,
                )
            case OptimizerType.ADAGRAD:
                optimizer = optim.Adagrad(
                    model.parameters(),
                    lr=self.optimizer_learning_rate,
                    weight_decay=self.optimizer_l2_regularization_strength,
                )
            case _:
                raise NotImplementedError()
        criterion = mse_with_l1_regularization(
            model,
            self.optimizer_l1_regularization_strength,
        )
        return (optimizer, criterion)
