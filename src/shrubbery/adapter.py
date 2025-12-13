import io
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
import torch.jit as jit
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class ModuleWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class TorchRegressor(BaseEstimator, RegressorMixin, ABC):
    def __init__(
        self,
        epochs: int,
        device: str,
    ) -> None:
        self.epochs = epochs
        self.device = device

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'TorchRegressor':
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        module, optimizer, criterion = self.prepare(input_dim=x.shape[1])
        module = ModuleWrapper(module).to(self.device)
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in (progress := tqdm(range(self.epochs))):
            module.train()
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = module(x_batch)
                metric = criterion(outputs.squeeze(), y_batch)
                metric.backward()
                optimizer.step()
            progress.set_description(
                f'Training - epoch: {epoch}; metric: {metric:.5f}'
            )
        self.serialized_model_ = io.BytesIO()
        jit.save(jit.script(module), self.serialized_model_)
        self.serialized_model_.seek(0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
        self.serialized_model_.seek(0)
        model = torch.jit.load(self.serialized_model_)
        self.serialized_model_.seek(0)
        model.eval()
        with torch.no_grad():
            predictions = model(x_tensor).cpu().numpy().squeeze()
        return predictions

    @abstractmethod
    def prepare(
        self, input_dim: int
    ) -> tuple[
        nn.Module,
        torch.optim.Optimizer,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        pass
