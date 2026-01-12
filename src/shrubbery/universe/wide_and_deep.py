# Code inspired by inspired by: https://github.com/Jeremy123W/Numerai
# Improved with help of: https://keras.io/examples/structured_data/wide_deep_cross_networks/  # noqa: E501
# As an alternative to the from-scratch approach one can also use:
# * https://github.com/jrzaurin/pytorch-widedeep
# Wide and Deep Learning research paper: https://arxiv.org/abs/1606.07792
# Wide & Deep Learning for RecSys with Pytorch: https://www.kaggle.com/code/matanivanov/wide-deep-learning-for-recsys-with-pytorch  # noqa: E501
import os
from enum import Enum
from typing import Callable

os.environ['KERAS_BACKEND'] = 'torch'

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from keras import Model  # noqa: E402
from keras.layers import (  # noqa: E402
    BatchNormalization,
    Concatenate,
    Dense,
    Dropout,
    Input,
    ReLU,
)
from keras.models import Sequential  # noqa: E402
from keras.ops import convert_to_tensor  # noqa: E402
from keras.optimizers import Adagrad, Adam  # noqa: E402
from sklearn.base import BaseEstimator, RegressorMixin  # noqa: E402

from shrubbery.adapter import TorchRegressor  # noqa: E402
from shrubbery.utilities import (  # noqa: E402
    deserialize_keras_model,
    serialize_keras_model,
)


class ModelType(str, Enum):
    WIDE = 'wide'
    DEEP = 'deep'
    WIDE_AND_DEEP = 'wide_and_deep'


class OptimizerType(str, Enum):
    SGD = 'sgd'
    ADAM = 'adam'
    ADAGRAD = 'adagrad'


class WideAndDeep(BaseEstimator, RegressorMixin):
    FIT_SHUFFLE = False

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
    ) -> None:
        self.model_type = model_type
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.serialized_model: bytes | None = None
        self.units = units
        self.optimizer_type = optimizer_type
        self.optimizer_learning_rate = optimizer_learning_rate
        self.optimizer_l1_regularization_strength = (
            optimizer_l1_regularization_strength
        )
        self.optimizer_l2_regularization_strength = (
            optimizer_l2_regularization_strength
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'WideAndDeep':
        model_columns = list(range(x.shape[1]))

        if self.optimizer_type == OptimizerType.SGD:
            optimizer = 'sgd'
        elif self.optimizer_type == OptimizerType.ADAM:
            optimizer = Adam(
                learning_rate=self.optimizer_learning_rate,
            )
        elif self.optimizer_type == OptimizerType.ADAGRAD:
            optimizer = Adagrad(
                learning_rate=self.optimizer_learning_rate,
                l1_regularization_strength=(
                    self.optimizer_l1_regularization_strength
                ),
                l2_regularization_strength=(
                    self.optimizer_l2_regularization_strength
                ),
            )

        model: Model
        model_input = Input(shape=(len(model_columns),))
        match self.model_type:
            case ModelType.WIDE:
                # Build a wide model using Sequential API
                # (linear regression with no activation)
                model = Sequential()
                model.add(model_input)
                model.add(BatchNormalization())
                model.add(Dense(1, activation='linear'))
            case ModelType.DEEP:
                # Build a deep model using Sequential API
                # (focusing solely on a deep neural network for regression)
                model = Sequential()
                model.add(model_input)
                for units in self.units:
                    model.add(Dense(units))
                    model.add(BatchNormalization())
                    model.add(ReLU())
                    model.add(Dropout(self.dropout_rate))
                model.add(
                    Dense(1, activation='linear')
                )  # Output layer, no activation for regression
            case ModelType.WIDE_AND_DEEP:
                # The Wide and Deep architecture is a specific neural network
                # architecture introduced by Google for handling a combination
                # of memorization
                # (shallow, wide paths for memorizing sparse features)
                # and generalization
                # (deep paths for learning abstract representations).
                #
                # Build a wide and deep model using Functional API:
                wide_input = deep_output = model_input
                # Define the deep path
                for units in self.units:
                    deep_output = Dense(units)(deep_output)
                    deep_output = BatchNormalization()(deep_output)
                    deep_output = ReLU()(deep_output)
                    deep_output = Dropout(self.dropout_rate)(deep_output)
                # Define the wide path (linear model)
                wide_output = BatchNormalization()(wide_input)
                # Concatenate deep and wide paths
                combined = Concatenate()([deep_output, wide_output])
                # Final output layer
                output = Dense(1, activation='linear')(combined)
                # Create the model
                model = Model(inputs=[model_input], outputs=output)
            case _:
                raise NotImplementedError()
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        model.fit(
            convert_to_tensor(x),
            convert_to_tensor(y),
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=WideAndDeep.FIT_SHUFFLE,
        )
        self.serialized_model_ = serialize_keras_model(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.serialized_model_ is not None
        model = deserialize_keras_model(self.serialized_model_)
        result = model.predict(convert_to_tensor(x))
        return result


class WideModule(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class DeepModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: list[int],
        dropout_rate: float,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for unit in units:
            layers.append(nn.Linear(input_dim, unit))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = unit
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class WideAndDeepModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        units: list[int],
        dropout_rate: float,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()
        # Deep path
        deep_layers: list[nn.Module] = []
        deep_input_dim = input_dim
        for unit in units:
            deep_layers.append(nn.Linear(deep_input_dim, unit))
            if use_batch_norm:
                deep_layers.append(nn.BatchNorm1d(unit))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.Dropout(dropout_rate))
            deep_input_dim = unit
        self.deep_network = nn.Sequential(*deep_layers)
        # Combined output: concat wide (raw input) + deep features, then project
        combined_dim = input_dim + deep_input_dim
        self.output_layer = nn.Linear(combined_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wide_output = x  # Wide path: raw features (no transformation)
        deep_output = self.deep_network(x)
        combined = torch.cat([wide_output, deep_output], dim=1)
        return self.output_layer(combined)


def mse_with_l1_l2_regularization(
    module: nn.Module, l1_scale: float, l2_scale: float, device: str
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    mse = nn.MSELoss()

    def regularized_loss(
        y_prediction: torch.Tensor, y_true: torch.Tensor
    ) -> torch.Tensor:
        l1_loss = torch.tensor(0.0).to(device)
        l2_loss = torch.tensor(0.0).to(device)
        for param in module.parameters():
            if param.ndim > 1:
                l1_loss += torch.sum(torch.abs(param))
                l2_loss += torch.sum(param**2)
        return mse(y_prediction, y_true) + l1_scale * l1_loss + l2_scale * l2_loss

    return regularized_loss


class WideAndDeepRegressor(TorchRegressor):
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
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, device=device)
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
        self.use_batch_norm = use_batch_norm

    def prepare(
        self, input_dim: int
    ) -> tuple[
        nn.Module,
        torch.optim.Optimizer,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        module: nn.Module
        match self.model_type:
            case ModelType.WIDE:
                module = WideModule(input_dim)
            case ModelType.DEEP:
                module = DeepModule(
                    input_dim,
                    self.units,
                    self.dropout_rate,
                    self.use_batch_norm,
                )
            case ModelType.WIDE_AND_DEEP:
                module = WideAndDeepModule(
                    input_dim,
                    self.units,
                    self.dropout_rate,
                    self.use_batch_norm,
                )
            case _:
                raise NotImplementedError()

        optimizer: torch.optim.Optimizer
        match self.optimizer_type:
            case OptimizerType.SGD:
                optimizer = optim.SGD(
                    module.parameters(), lr=self.optimizer_learning_rate
                )
            case OptimizerType.ADAM:
                optimizer = optim.Adam(
                    module.parameters(), lr=self.optimizer_learning_rate
                )
            case OptimizerType.ADAGRAD:
                optimizer = optim.Adagrad(
                    module.parameters(), lr=self.optimizer_learning_rate
                )
            case _:
                raise NotImplementedError()

        criterion = mse_with_l1_l2_regularization(
            module,
            self.optimizer_l1_regularization_strength,
            self.optimizer_l2_regularization_strength,
            self.device,
        )

        return (module, optimizer, criterion)
