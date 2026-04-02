import math

from openTSNE import TSNE
from sklearn.base import BaseEstimator, TransformerMixin


class OpenTSNE(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        estimator: TSNE,
    ) -> None:
        self.estimator = estimator

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'OpenTSNE':
        self.embedder_ = self.estimator.fit(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        assert self.embedder_ is not None
        return self.embedder_.transform(x)
