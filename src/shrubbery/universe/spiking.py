import numpy as np
import snntorch as snn
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_X_y
from skorch import NeuralNetRegressor


class SpikingModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        beta: float,
        with_surrogate_gradients: bool,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.lif1 = (
            snn.Leaky(beta=beta, spike_grad=snn.surrogate.fast_sigmoid())
            if with_surrogate_gradients
            else snn.Leaky(beta=beta)
        )
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        mem1 = torch.zeros(x.size(0), self.fc1.out_features, device=x.device)
        spk1, _ = self.lif1(x, mem1)
        x = self.fc2(spk1)
        return x


class SpikingRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        hidden_dim: int,
        beta: float,
        with_surrogate_gradients: bool,
        lr: float,
        batch_size: int,
        epochs: int,
        device: str | None,
        random_state: int,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.with_surrogate_gradients = with_surrogate_gradients
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
        x_tensor = torch.from_numpy(x).to(torch.float32).to(self.device)
        y_tensor = (
            torch.from_numpy(y).to(torch.float32).to(self.device).unsqueeze(1)
        )
        # Build model
        module = SpikingModule(
            input_dim=x.shape[1],
            hidden_dim=self.hidden_dim,
            output_dim=1,
            beta=self.beta,
            with_surrogate_gradients=self.with_surrogate_gradients,
        ).to(self.device)
        # Train model
        self.model_ = NeuralNetRegressor(
            module,
            criterion=torch.nn.MSELoss,
            optimizer=torch.optim.Adam,
            lr=self.lr,
            batch_size=self.batch_size,
            max_epochs=self.epochs,
            train_split=None,
            device=self.device,
        )
        self.model_.fit(x_tensor, y_tensor)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = check_array(x)
        x_tensor = torch.from_numpy(x).to(torch.float32).to(self.device)
        return self.model_.predict(x_tensor)
