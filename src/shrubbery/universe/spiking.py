import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y
from torch.utils.data import DataLoader, TensorDataset

from shrubbery.observability import logger


class SpikingRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_dim: int,
        beta: float,
        with_surrogate_gradients: bool,
        num_steps: int,
        lr: float,
        batch_size: int,
        epochs: int,
        device: str | None,
        random_state: int,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.with_surrogate_gradients = with_surrogate_gradients
        self.num_steps = num_steps
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        torch.manual_seed(self.random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'SpikingRegressor':
        x, y = check_X_y(x, y, y_numeric=True)
        x_tensor = torch.from_numpy(x).to(torch.float32)
        y_tensor = torch.from_numpy(y).to(torch.float32).unsqueeze(1)
        # Build model
        input_dim = x.shape[1]
        output_dim = 1
        fc1 = nn.Linear(input_dim, self.hidden_dim)
        if self.with_surrogate_gradients:
            lif1 = snn.Leaky(
                beta=self.beta, spike_grad=snn.surrogate.fast_sigmoid()
            )
        else:
            lif1 = snn.Leaky(beta=self.beta)
        fc2 = nn.Linear(self.hidden_dim, output_dim)
        model = nn.Sequential(fc1, lif1, fc2).to(self.device)
        # Train model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                spk_rec = []
                mem = lif1.init_leaky()
                for _ in range(self.num_steps):
                    cur = fc1(xb)
                    spk, mem = lif1(cur, mem)
                    out = fc2(spk)
                    spk_rec.append(out)
                out_rec = torch.stack(spk_rec).mean(0)
                loss = criterion(out_rec, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(
                f'Epoch {epoch + 1}/{self.epochs} - Loss: {total_loss / len(loader):.6f}'
            )
        self.scripted_model_ = torch.jit.script(model)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = check_array(x)
        x_tensor = torch.from_numpy(x).to(torch.float32).to(self.device)
        self.scripted_model_.eval()
        with torch.no_grad():
            predictions = self.scripted_model_(x_tensor)
        return predictions.cpu().numpy().ravel()
