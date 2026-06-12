"""Save and load trained model artifacts with joblib.

An artifact bundles the fitted estimator plus the metadata needed to reproduce a
prediction: the exact feature column order, target, horizon, and training window.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib

ARTIFACT_DIR = Path("models_store")


@dataclass
class ModelArtifact:
    """A trained model plus everything needed to apply it consistently."""

    estimator: object
    model_name: str
    target: str
    horizon: int
    feature_names: list[str]
    ticker: str
    trained_at: str = field(default_factory=lambda: dt.datetime.utcnow().isoformat())
    metrics: dict = field(default_factory=dict)

    def metadata(self) -> dict:
        """Serializable metadata (excludes the estimator object)."""
        d = asdict(self)
        d.pop("estimator", None)
        return d


def artifact_path(ticker: str, model_name: str, target: str) -> Path:
    return ARTIFACT_DIR / f"{ticker.upper()}__{model_name}__{target}.joblib"


def save_model(artifact: ModelArtifact) -> Path:
    """Persist an artifact to disk and return its path."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = artifact_path(artifact.ticker, artifact.model_name, artifact.target)
    joblib.dump(artifact, path)
    return path


def load_model(path: str | Path) -> ModelArtifact:
    """Load an artifact from disk."""
    return joblib.load(Path(path))
