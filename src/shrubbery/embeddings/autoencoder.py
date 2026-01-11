# Code inspired by: https://github.com/jimfleming/numerai/blob/master/models/autoencoder/model.py  # noqa: E501
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import DataLoader, TensorDataset


class AutoencoderNetwork(nn.Module):
    """PyTorch autoencoder network module."""

    def __init__(self, input_dim: int, layer_units: list[int]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.layer_units = layer_units

        # Build encoder
        encoder_layers = []
        prev_units = input_dim
        for units in layer_units:
            encoder_layers.extend(
                [
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units, eps=1e-5),
                    nn.Sigmoid(),
                ]
            )
            prev_units = units

        # Build decoder (symmetric)
        decoder_layer_units = layer_units[:-1][::-1] + [input_dim]
        decoder_layers = []
        for units in decoder_layer_units:
            decoder_layers.extend(
                [
                    nn.Linear(prev_units, units),
                    nn.BatchNorm1d(units, eps=1e-5),
                    nn.Sigmoid(),
                ]
            )
            prev_units = units

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights with VarianceScaling equivalent
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights using truncated normal (VarianceScaling equivalent)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # VarianceScaling with mode='fan_in', scale=1.0
                # Equivalent to Kaiming normal with a=0
                fan_in = module.weight.size(1)
                std = (1.0 / fan_in) ** 0.5
                # Truncate at 2 standard deviations
                init.trunc_normal_(
                    module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
                )
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoded representation."""
        return self.encoder(x)


class Autoencoder(BaseEstimator, TransformerMixin):
    """sklearn-compatible PyTorch autoencoder embedder."""

    def __init__(
        self,
        batch_size: int,
        epochs: int,
        layer_units: list[int],
        denoise: bool,
        learning_rate: float,
    ) -> None:
        self.batch_size = batch_size
        self.epochs = epochs
        self.layer_units = layer_units
        self.denoise = denoise
        self.learning_rate = learning_rate

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'Autoencoder':
        """Fit the autoencoder to the data."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build model
        input_dim = x.shape[1]
        model = AutoencoderNetwork(input_dim, self.layer_units).to(device)

        # Apply denoising if requested
        x_train = x.copy()
        if self.denoise:
            x_variance = np.var(x, axis=0)
            x_stddev = np.sqrt(x_variance)
            noise = np.random.normal(0, 0.1 * x_stddev, size=x.shape)
            x_train = x_train + noise

        # Create data loader
        x_tensor = torch.FloatTensor(x_train).to(device)
        x_target = torch.FloatTensor(x).to(device)
        dataset = TensorDataset(x_tensor, x_target)
        dataloader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        # Setup optimizer with L2 regularization (weight_decay)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-3,  # L2 regularization
        )

        # Gradient clipping will be applied manually
        criterion = nn.MSELoss()

        # Training loop
        model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_target in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_target)
                loss.backward()

                # Apply gradient clipping (clipnorm=1.0)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

        # Store model configuration and weights
        self.input_dim_ = input_dim
        self.encoder_state_dict_ = model.encoder.state_dict()
        self.device_ = 'cuda' if torch.cuda.is_available() else 'cpu'

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Transform data using the trained encoder."""
        assert hasattr(self, 'encoder_state_dict_'), 'Model not fitted yet'

        device = torch.device(self.device_)

        # Reconstruct encoder
        full_model = AutoencoderNetwork(self.input_dim_, self.layer_units).to(
            device
        )
        full_model.encoder.load_state_dict(self.encoder_state_dict_)
        full_model.encoder.eval()

        # Transform data
        x_tensor = torch.FloatTensor(x).to(device)
        with torch.no_grad():
            encoded = full_model.encode(x_tensor)
            result = encoded.cpu().numpy()

        return result
