import io
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch_tensorrt
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import train_test_split
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class CompilerBackend(str, Enum):
    TENSORRT = 'tensorrt'
    INDUCTOR = 'inductor'
    JIT = 'jit'


class SchedulerType(str, Enum):
    COSINE = 'cosine'
    ONE_CYCLE = 'one_cycle'


@dataclass(frozen=True)
class LearningSchedule:
    scheduler: SchedulerType
    warmup_epochs: int = 0
    warmup_start_factor: float = 0.1
    cosine_min_lr_ratio: float = 0.01
    one_cycle_pct_start: float = 0.1


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: LearningSchedule | None,
    learning_rate: float,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """
    Build an LR scheduler from a LearningSchedule config, stepped once per epoch.
    Returns None when no scheduling is requested.
    """
    if schedule is None:
        return None
    main: torch.optim.lr_scheduler.LRScheduler | None = None
    match schedule.scheduler:
        case SchedulerType.ONE_CYCLE:
            if schedule.warmup_epochs > 0:
                raise ValueError(
                    'ONE_CYCLE has built-in warmup via pct_start; warmup_epochs must be 0'
                )
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=learning_rate,
                total_steps=epochs,
                pct_start=schedule.one_cycle_pct_start,
            )
        case SchedulerType.COSINE:
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - schedule.warmup_epochs,
                eta_min=learning_rate * schedule.cosine_min_lr_ratio,
            )
    if schedule.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=schedule.warmup_start_factor,
            total_iters=schedule.warmup_epochs,
        )
        if main is not None:
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, main],
                milestones=[schedule.warmup_epochs],
            )
        return warmup
    return main


@dataclass(frozen=True)
class EarlyStopping:
    val_fraction: float = 0.1
    patience: int = 5
    min_delta: float = 0.0
    min_epochs: int = 1


class EarlyStoppingState:
    def __init__(self, config: EarlyStopping) -> None:
        self._config = config
        self._best_loss = float('inf')
        self._epochs_without_improvement = 0

    def step(
        self,
        epoch: int,
        validation_loss: float,
    ) -> bool:
        """Update state; returns True when training should stop."""
        config = self._config
        if validation_loss < self._best_loss - config.min_delta:
            self._best_loss = validation_loss
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
        if (
            epoch + 1 >= config.min_epochs
            and self._epochs_without_improvement >= config.patience
        ):
            return True
        return False


class ModelWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class TorchEstimator(BaseEstimator, TransformerMixin, RegressorMixin):
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        device: str,
        compiler: CompilerBackend = CompilerBackend.JIT,
        autocast: bool = False,
        learning_schedule: LearningSchedule | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.compiler = compiler
        self.autocast = autocast
        self.learning_schedule = learning_schedule
        self.early_stopping = early_stopping

    def autocast_context(self) -> AbstractContextManager:
        if self.autocast:
            return autocast(device_type=self.device, dtype=torch.bfloat16)
        return nullcontext()

    def train(self, x: torch.Tensor, y: torch.Tensor) -> nn.Module:
        x_training, y_training, x_validation, y_validation = x, y, None, None
        if self.early_stopping is not None:
            x_training, x_validation, y_training, y_validation = (
                train_test_split(
                    x,
                    y,
                    test_size=self.early_stopping.val_fraction,
                    shuffle=False,
                )
            )
        module = self.module(input_dim=x_training.shape[1])
        model = ModelWrapper(module).to(self.device)
        dataset = TensorDataset(x_training, y_training)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer, criterion = self.prepare(model)
        scheduler = make_scheduler(
            optimizer, self.learning_schedule, self.learning_rate, self.epochs
        )
        early_stopping_state = (
            EarlyStoppingState(self.early_stopping)
            if self.early_stopping is not None
            else None
        )
        for epoch in range(self.epochs):
            model.train()
            for x_batch, y_batch in (progress := tqdm(loader)):
                optimizer.zero_grad()
                with self.autocast_context():
                    outputs = model(x_batch)
                    metric = criterion(outputs.squeeze(), y_batch)
                metric.backward()
                optimizer.step()
                progress.set_description(
                    f'Training - epoch: {epoch}; metric: {metric:.4f}'
                )
            if scheduler is not None:
                scheduler.step()
            if (
                early_stopping_state is not None
                and x_validation is not None
                and y_validation is not None
            ):
                model.eval()
                with torch.no_grad():
                    with self.autocast_context():
                        validation_output = model(x_validation)
                        validation_loss = criterion(
                            validation_output.squeeze(), y_validation
                        ).item()
                if early_stopping_state.step(epoch, validation_loss):
                    break
        return model

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'TorchEstimator':
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        module = self.train(x_tensor, y_tensor)
        self.serialized_model_ = io.BytesIO()
        torch.save(module.state_dict(), self.serialized_model_)
        self.serialized_model_.seek(0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        self.serialized_model_.seek(0)
        module = self.module(input_dim=x.shape[1])
        model = ModelWrapper(module)
        # Load only weights (avoid executing arbitrary pickle code risk)
        model.load_state_dict(
            torch.load(self.serialized_model_, weights_only=True)
        )
        self.serialized_model_.seek(0)
        model.eval().to(self.device)
        match self.compiler:
            case CompilerBackend.TENSORRT:
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                model = torch_tensorrt.compile(
                    model.to(torch.float32),
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=(1, x.shape[1]),
                            opt_shape=x.shape,
                            max_shape=x.shape,
                            dtype=torch.float32,
                        )
                    ],
                    optimization_level=5,
                )
            case CompilerBackend.INDUCTOR:
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                model = torch.compile(
                    model,
                    backend='inductor',
                    mode='max-autotune',
                    dynamic=True,
                )
            case CompilerBackend.JIT:
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
                model = torch.jit.script(model)
        with torch.inference_mode():
            result = model(x_tensor)
        return result.cpu().numpy().squeeze()

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)

    def module(self, input_dim: int) -> nn.Module:
        raise NotImplementedError('TorchEstimator.module not implemented')

    def prepare(
        self, model: nn.Module
    ) -> tuple[
        torch.optim.Optimizer,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        raise NotImplementedError('TorchEstimator.module not implemented')


def variance_scaling_initializer_with_fan_in(module: nn.Module) -> None:
    """Initialize weights using variance scaling (with fan-in and factor of 1.0)."""
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            fan_in = submodule.weight.size(1)
            std = (1.0 / fan_in) ** 0.5
            init.trunc_normal_(
                submodule.weight, mean=0.0, std=std, a=-2 * std, b=2 * std
            )
            if submodule.bias is not None:
                init.zeros_(submodule.bias)
