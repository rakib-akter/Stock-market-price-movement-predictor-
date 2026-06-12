"""Machine-learning models: registry, training, prediction, persistence."""

from backend.app.models.persistence import load_model, save_model
from backend.app.models.predict import predict_latest, predict_proba_frame
from backend.app.models.registry import available_models, build_estimator
from backend.app.models.train import train_classifier

__all__ = [
    "available_models",
    "build_estimator",
    "train_classifier",
    "predict_latest",
    "predict_proba_frame",
    "save_model",
    "load_model",
]
