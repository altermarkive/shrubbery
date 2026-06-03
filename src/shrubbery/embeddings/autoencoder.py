# Code inspired by: https://github.com/jimfleming/numerai/blob/master/models/autoencoder/model.py  # noqa: E501
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from shrubbery.adapter import (
    CompilerBackend,
    EarlyStopping,
    EarlyStoppingState,
    LearningSchedule,
    ModelWrapper,
    TorchEstimator,
    make_scheduler,
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


class AutoencoderEmbedder(TorchEstimator):
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        layer_units: list[int],
        denoise: bool,
        learning_rate: float,
        batch_norm_eps: float,
        device: str,
        compiler: CompilerBackend,
        learning_schedule: LearningSchedule | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            compiler=compiler,
            learning_schedule=learning_schedule,
            early_stopping=early_stopping,
        )
        self.layer_units = layer_units
        self.denoise = denoise
        self.batch_norm_eps = batch_norm_eps

    def train(self, x: torch.Tensor, y: torch.Tensor) -> nn.Module:
        x_training, x_validation = x, None
        if self.early_stopping is not None:
            x_training, x_validation = train_test_split(
                x, test_size=self.early_stopping.val_fraction, shuffle=False
            )
        # Autoencoder
        input_dim = x_training.shape[1]
        module = AutoencoderNetwork(
            input_dim, self.layer_units, self.batch_norm_eps
        ).to(self.device)
        # Training
        if self.denoise:
            x_stddev = x_training.var(dim=0).sqrt()
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-3,
        )
        criterion = nn.MSELoss()
        scheduler = make_scheduler(
            optimizer, self.learning_schedule, self.learning_rate, self.epochs
        )
        early_stopping_state = (
            EarlyStoppingState(self.early_stopping)
            if self.early_stopping is not None
            else None
        )
        for epoch in range(self.epochs):
            x_training = (
                x_training + torch.randn_like(x_training) * (0.1 * x_stddev)
                if self.denoise
                else x_training
            )
            dataset = TensorDataset(x_training.to(self.device))
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )
            module.train()
            metric_sum = 0.0
            for i, (x_batch,) in enumerate(progress := tqdm(loader)):
                optimizer.zero_grad()
                with autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = module(x_batch)
                    metric = criterion(outputs, x_batch)
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
            if scheduler is not None:
                scheduler.step()
            if early_stopping_state is not None and x_validation is not None:
                module.eval()
                with torch.no_grad():
                    with autocast(
                        device_type=self.device, dtype=torch.bfloat16
                    ):
                        validation_loss = criterion(
                            module(x_validation), x_validation
                        ).item()
                if early_stopping_state.step(epoch, validation_loss):
                    break
        return ModelWrapper(module.encoder)

    def module(self, input_dim: int) -> nn.Module:
        return AutoencoderNetwork(
            input_dim, self.layer_units, self.batch_norm_eps
        ).encoder
