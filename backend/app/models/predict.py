"""Apply a trained artifact to produce predictions and confidence scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.models.persistence import ModelArtifact


def _align_features(artifact: ModelArtifact, matrix: pd.DataFrame) -> pd.DataFrame:
    """Select and order columns exactly as the model saw during training."""
    missing = [c for c in artifact.feature_names if c not in matrix.columns]
    if missing:
        raise ValueError(f"Feature matrix is missing trained columns: {missing}")
    return matrix[artifact.feature_names].astype(float)


def predict_proba_frame(
    artifact: ModelArtifact, matrix: pd.DataFrame
) -> pd.DataFrame:
    """Return a frame indexed by date with predicted class, prob, and confidence.

    ``confidence`` is ``|p_up - 0.5| * 2`` ∈ [0, 1] — distance from a coin flip.
    Remember: this is a model-internal score, **not** a calibrated edge.
    """
    X = _align_features(artifact, matrix)
    est = artifact.estimator

    if hasattr(est, "predict_proba"):
        p_up = est.predict_proba(X)[:, 1]
    else:  # fall back to decision_function squashed to [0,1]
        scores = est.decision_function(X)
        p_up = 1.0 / (1.0 + np.exp(-scores))

    pred_class = (p_up >= 0.5).astype(int)
    confidence = np.abs(p_up - 0.5) * 2.0

    return pd.DataFrame(
        {
            "prob_up": p_up,
            "predicted_class": pred_class,
            "confidence": confidence,
        },
        index=X.index,
    )


def predict_latest(artifact: ModelArtifact, matrix: pd.DataFrame) -> dict:
    """Return the most recent row's prediction as a plain dict (for the API)."""
    frame = predict_proba_frame(artifact, matrix)
    last = frame.iloc[-1]
    return {
        "ticker": artifact.ticker,
        "date": str(frame.index[-1].date() if hasattr(frame.index[-1], "date") else frame.index[-1]),
        "model": artifact.model_name,
        "horizon": artifact.horizon,
        "prob_up": float(last["prob_up"]),
        "predicted_class": int(last["predicted_class"]),
        "direction": "UP" if last["predicted_class"] == 1 else "DOWN",
        "confidence": float(last["confidence"]),
    }
