from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim

from shrubbery.adapter import TorchRegressor


class CNNModule(nn.Module):
    """
    CNN for tabular data - reshapes 1D features into 2D grid.
    Designed for Numerai's 42-feature dataset (reshaped to 6x7).
    """

    def __init__(
        self,
        input_dim: int,
        num_filters: list[int],
        kernel_sizes: list[int],
        dense_units: list[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim

        # Calculate reshape dimensions (closest to square)
        # For 42 features: 6x7 = 42
        self.height = int(input_dim**0.5)
        while input_dim % self.height != 0:
            self.height -= 1
        self.width = input_dim // self.height

        # Convolutional layers
        conv_layers: list[nn.Module] = []
        in_channels = 1  # Single channel for tabular data

        for i, (filters, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            # Ensure kernel size doesn't exceed dimensions
            ks = min(kernel_size, min(self.height, self.width))
            padding = ks // 2  # Same padding

            conv_layers.append(
                nn.Conv2d(
                    in_channels,
                    filters,
                    kernel_size=ks,
                    padding=padding,
                    bias=False,
                )
            )
            conv_layers.append(nn.BatchNorm2d(filters))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout2d(dropout_rate))

            # Add max pooling for dimensionality reduction (not on last layer)
            if i < len(num_filters) - 1:
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            in_channels = filters

        self.conv_block = nn.Sequential(*conv_layers)

        # Calculate flattened dimension after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.height, self.width)
            conv_output = self.conv_block(dummy_input)
            flattened_dim = conv_output.view(1, -1).shape[1]

        # Dense layers
        dense_layers: list[nn.Module] = []
        prev_units = flattened_dim

        for units in dense_units:
            dense_layers.append(nn.Linear(prev_units, units))
            dense_layers.append(nn.BatchNorm1d(units))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout_rate))
            prev_units = units

        # Output layer
        dense_layers.append(nn.Linear(prev_units, 1))
        dense_layers.append(nn.Sigmoid())

        self.dense_block = nn.Sequential(*dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape from (batch, features) to (batch, 1, height, width)
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, self.height, self.width)

        # Apply convolutions
        x = self.conv_block(x)

        # Flatten
        x = x.view(batch_size, -1)

        # Apply dense layers
        x = self.dense_block(x)

        return x


class CNNRegressor(TorchRegressor):
    """
    Convolutional Neural Network Regressor for tabular data.

    Reshapes 1D feature vectors into 2D grids and applies convolutional
    layers followed by dense layers for regression. Suitable for ensemble
    use with Numerai's small dataset (42 features).

    Parameters
    ----------
    num_filters : list[int]
        Number of filters for each convolutional layer.
        Example: [32, 64, 128] creates 3 conv layers.
    kernel_sizes : list[int]
        Kernel size for each convolutional layer.
        Example: [3, 3, 3] uses 3x3 kernels for all layers.
    dense_units : list[int]
        Number of units in each dense layer after convolutions.
        Example: [128, 64] creates 2 dense layers.
    dropout_rate : float
        Dropout rate for regularization (applied to both conv and dense layers).
    learning_rate : float
        Learning rate for optimizer.
    weight_decay : float
        L2 regularization weight decay for AdamW optimizer.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    device : str
        Device to use for training ('cpu' or 'cuda').
    """

    def __init__(
        self,
        num_filters: list[int],
        kernel_sizes: list[int],
        dense_units: list[int],
        dropout_rate: float,
        learning_rate: float,
        weight_decay: float,
        epochs: int,
        batch_size: int,
        device: str,
    ) -> None:
        super().__init__(epochs=epochs, batch_size=batch_size, device=device)
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
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
        # Build CNN module
        module = CNNModule(
            input_dim=input_dim,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            dense_units=self.dense_units,
            dropout_rate=self.dropout_rate,
        )

        # Optimizer with weight decay (L2 regularization)
        optimizer = optim.AdamW(
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Binary cross-entropy loss (suitable for Numerai targets in [0, 1])
        criterion = nn.BCELoss()

        return (module, optimizer, criterion)
