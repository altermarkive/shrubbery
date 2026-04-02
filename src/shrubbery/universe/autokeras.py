import tempfile
from pathlib import Path

import autokeras as ak
import numpy as np
from keras import Model
from keras.ops import convert_to_tensor
from keras.saving import load_model as load_keras_model
from keras.saving import save_model as save_keras_model
from sklearn.base import BaseEstimator, RegressorMixin


def serialize_keras_model(model: Model) -> bytes:
    with tempfile.NamedTemporaryFile(
        suffix='.keras', delete_on_close=False
    ) as handle:
        handle.close()
        path = Path(handle.name)
        save_keras_model(model, path)
        return path.read_bytes()


def deserialize_keras_model(model: bytes) -> Model:
    with tempfile.NamedTemporaryFile(
        suffix='.keras', delete_on_close=False
    ) as handle:
        handle.close()
        path = Path(handle.name)
        path.write_bytes(model)
        return load_keras_model(path, custom_objects=ak.CUSTOM_OBJECTS)


class AutoKerasRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        max_trials: int,
        epochs: int,
    ) -> None:
        self.max_trials = max_trials
        self.epochs = epochs

    def fit(self, x: np.ndarray, y: np.ndarray) -> 'AutoKerasRegressor':
        model = ak.StructuredDataRegressor(
            column_names=[f'feature{i}' for i in range(x.shape[1])],
            column_types={
                f'feature{i}': 'numerical' for i in range(x.shape[1])
            },
            max_trials=self.max_trials,
            overwrite=True,
        )
        model.fit(
            convert_to_tensor(x),
            convert_to_tensor(y),
            epochs=self.epochs,
        )
        self.serialized_model_ = serialize_keras_model(model.export_model())
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        assert self.serialized_model_ is not None
        model = deserialize_keras_model(self.serialized_model_)
        result = model.predict(convert_to_tensor(x))
        return result
