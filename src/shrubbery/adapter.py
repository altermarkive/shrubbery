import io
from enum import Enum
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch_tensorrt
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class CompilerBackend(str, Enum):
    TENSORRT = 'tensorrt'
    INDUCTOR = 'inductor'
    JIT = 'jit'


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
        device: str,
        compiler: CompilerBackend = CompilerBackend.TENSORRT,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.compiler = compiler

    def train(self, x: torch.Tensor, y: torch.Tensor) -> nn.Module:
        module = self.module(input_dim=x.shape[1])
        model = ModelWrapper(module).to(self.device)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer, criterion = self.prepare(model)
        for epoch in range(self.epochs):
            model.train()
            for x_batch, y_batch in (progress := tqdm(loader)):
                optimizer.zero_grad()
                with autocast(device_type=self.device, dtype=torch.bfloat16):
                    outputs = model(x_batch)
                    metric = criterion(outputs.squeeze(), y_batch)
                metric.backward()
                optimizer.step()
                progress.set_description(
                    f'Training - epoch: {epoch}; metric: {metric:.4f}'
                )
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
                x_tensor = torch.tensor(x, dtype=torch.bfloat16).to(
                    self.device
                )
                model = torch_tensorrt.compile(
                    model.to(torch.bfloat16),
                    inputs=[
                        torch_tensorrt.Input(
                            min_shape=(1, x.shape[1]),
                            opt_shape=x.shape,
                            max_shape=x.shape,
                            dtype=torch.bfloat16,
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
