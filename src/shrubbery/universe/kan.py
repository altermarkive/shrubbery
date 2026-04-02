import torch
from kan import KAN
from kan.utils import create_dataset_from_data
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin


class KANRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        width: list[int] = [],
        grid: int = 3,
        k: int = 3,
        mult_arity: int | list[int] = 2,
        noise_scale: float = 0.3,
        scale_base_mu: float = 0.0,
        scale_base_sigma: float = 1.0,
        base_fun: str = 'silu',
        symbolic_enabled: bool = True,
        affine_trainable: bool = False,
        grid_eps: float = 0.02,
        grid_range: list[int] = [-1, 1],
        sp_trainable: bool = True,
        sb_trainable: bool = True,
        seed: int = 1,
        save_act: bool = True,
        sparse_init: bool = False,
        auto_save: bool = True,
        first_init: bool = True,
        ckpt_path: str = './model',
        state_id: int = 0,
        round: int = 0,
        device: str = 'cpu',
    ) -> None:
        self.width = width
        self.grid = grid
        self.k = k
        self.mult_arity = mult_arity
        self.noise_scale = noise_scale
        self.scale_base_mu = scale_base_mu
        self.scale_base_sigma = scale_base_sigma
        self.base_fun = base_fun
        self.symbolic_enabled = symbolic_enabled
        self.affine_trainable = affine_trainable
        self.grid_eps = grid_eps
        self.grid_range = grid_range
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable
        self.seed = seed
        self.save_act = save_act
        self.sparse_init = sparse_init
        self.auto_save = auto_save
        self.first_init = first_init
        self.ckpt_path = ckpt_path
        self.state_id = state_id
        self.round = round
        self.device = device

    def fit(self, x: NDArray, y: NDArray) -> 'KANRegressor':
        self.model_ = KAN(
            width=self.width,
            grid=self.grid,
            k=self.k,
            mult_arity=self.mult_arity,
            noise_scale=self.noise_scale,
            scale_base_mu=self.scale_base_mu,
            scale_base_sigma=self.scale_base_sigma,
            base_fun=self.base_fun,
            symbolic_enabled=self.symbolic_enabled,
            affine_trainable=self.affine_trainable,
            grid_eps=self.grid_eps,
            grid_range=self.grid_range,
            sp_trainable=self.sp_trainable,
            sb_trainable=self.sb_trainable,
            seed=self.seed,
            save_act=self.save_act,
            sparse_init=self.sparse_init,
            auto_save=self.auto_save,
            first_init=self.first_init,
            ckpt_path=self.ckpt_path,
            state_id=self.state_id,
            round=self.round,
            device=self.device,
        )
        self.model_.fit(
            create_dataset_from_data(
                torch.from_numpy(x).to(self.device),
                torch.from_numpy(y).to(self.device),
            ),
        )
        return self

    def predict(self, x: NDArray) -> NDArray:
        assert self.model_ is not None
        predictions = self.model_(x)
        return predictions.detach().cpu().numpy().flatten()
