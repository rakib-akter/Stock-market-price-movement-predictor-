"""Machine-learning models: registry, training, prediction, persistence."""

from backend.app.models.calibration import brier_score, calibrate_estimator
from backend.app.models.persistence import load_model, save_model
from backend.app.models.predict import predict_latest, predict_proba_frame
from backend.app.models.registry import available_models, build_estimator
from backend.app.models.train import train_classifier
from backend.app.models.tuning import tune_hyperparameters

# NOTE: ``evaluate.compare_models`` lives a layer up (it depends on the backtester),
# so it is intentionally NOT eagerly imported here to avoid a models<->backtesting
# import cycle. Import it directly: ``from backend.app.models.evaluate import ...``.

__all__ = [
    "available_models",
    "build_estimator",
    "train_classifier",
    "predict_latest",
    "predict_proba_frame",
    "save_model",
    "load_model",
    "calibrate_estimator",
    "brier_score",
    "tune_hyperparameters",
]
