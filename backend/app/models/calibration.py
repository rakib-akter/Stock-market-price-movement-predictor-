"""Probability calibration.

A classifier's ``predict_proba`` is an internal score, not a true probability. A
model can have good ROC-AUC (ranking) but badly miscalibrated probabilities, which
matters the moment you size positions by confidence. We calibrate with a
**time-series-safe** cross-validation (``TimeSeriesSplit``) so calibration never
peeks at the future — using the default shuffled CV here would be leakage.

Quality is measured by the Brier score (mean squared error of probabilities;
lower is better) and, optionally, a reliability table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from backend.app.models.registry import build_estimator


def calibrate_estimator(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "isotonic",
    n_splits: int = 5,
) -> CalibratedClassifierCV:
    """Fit a calibrated classifier using time-ordered CV folds.

    ``method`` is ``"isotonic"`` (flexible, needs more data) or ``"sigmoid"``
    (Platt scaling, robust on small samples).
    """
    base = build_estimator(model_name)
    cv = TimeSeriesSplit(n_splits=n_splits)
    calibrated = CalibratedClassifierCV(base, method=method, cv=cv)
    calibrated.fit(X, y.astype(int))
    return calibrated


def reliability_table(
    y_true: pd.Series, prob: pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Bin predictions and compare predicted vs. observed up-rate per bin.

    A well-calibrated model has ``mean_predicted`` ≈ ``observed_rate`` in each bin.
    """
    df = pd.DataFrame({"y": y_true.astype(int).to_numpy(), "p": np.asarray(prob)})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    grouped = df.groupby("bin", observed=True).agg(
        mean_predicted=("p", "mean"),
        observed_rate=("y", "mean"),
        count=("y", "size"),
    )
    return grouped.reset_index(drop=True)


def brier_score(y_true: pd.Series, prob: pd.Series) -> float:
    """Brier score (lower is better; 0.25 is the no-skill 50/50 baseline)."""
    return float(brier_score_loss(y_true.astype(int), np.asarray(prob)))
