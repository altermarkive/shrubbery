from typing import Any


def about(
    model: Any,
) -> str:  # To be applied to NumeraiBestGridSearchEstimator
    feature_importance = model.estimator.estimator.get_booster().get_fscore()
    return str(feature_importance)
