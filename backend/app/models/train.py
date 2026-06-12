"""Training and evaluation for the classification target (next-N-day direction).

Evaluation is **out-of-sample on a time-ordered holdout**, and every metric is
reported next to a naive baseline so an "impressive" number can be sanity-checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.app.config import settings
from backend.app.data.splits import time_train_test_split
from backend.app.features.pipeline import feature_columns, split_X_y
from backend.app.models.persistence import ModelArtifact
from backend.app.models.registry import build_estimator
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainResult:
    artifact: ModelArtifact
    metrics: dict
    test_index: pd.Index
    y_test: pd.Series
    proba_test: np.ndarray


def _classification_metrics(y_true: pd.Series, y_pred, proba) -> dict:
    y_true = y_true.astype(int)
    # Majority-class baseline: predict whichever direction was more common in train-like data.
    majority = int(round(y_true.mean()))
    baseline_acc = max(y_true.mean(), 1 - y_true.mean())
    metrics = {
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "baseline_accuracy": float(baseline_acc),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_true.mean()),
        "majority_class": majority,
    }
    # ROC-AUC needs both classes present and probability scores.
    if proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
    return metrics


def train_classifier(
    matrix: pd.DataFrame,
    ticker: str,
    model_name: str = "logistic",
    horizon: int = 1,
    test_size: float | None = None,
) -> TrainResult:
    """Train a direction classifier on a built feature matrix.

    Parameters
    ----------
    matrix:
        Output of :func:`backend.app.features.pipeline.build_feature_matrix`.
    ticker:
        Symbol (stored in the artifact for traceability).
    model_name:
        Registry name (``logistic``, ``random_forest``, …).
    horizon:
        Label horizon; selects target column ``y_dir_{horizon}``.
    """
    target = f"y_dir_{horizon}"
    test_size = settings.test_size if test_size is None else test_size

    train_df, test_df = time_train_test_split(matrix, test_size=test_size)
    X_train, y_train = split_X_y(train_df, target)
    X_test, y_test = split_X_y(test_df, target)
    y_train, y_test = y_train.astype(int), y_test.astype(int)

    logger.info(
        "Training %s on %s | train=%d test=%d features=%d",
        model_name, ticker, len(X_train), len(X_test), X_train.shape[1],
    )

    estimator = build_estimator(model_name)
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_test)
    proba = (
        estimator.predict_proba(X_test)[:, 1]
        if hasattr(estimator, "predict_proba")
        else None
    )

    metrics = _classification_metrics(y_test, y_pred, proba)
    logger.info(
        "%s/%s  acc=%.3f (baseline %.3f)  auc=%s",
        ticker, model_name, metrics["accuracy"], metrics["baseline_accuracy"],
        f"{metrics.get('roc_auc'):.3f}" if "roc_auc" in metrics else "n/a",
    )

    artifact = ModelArtifact(
        estimator=estimator,
        model_name=model_name,
        target=target,
        horizon=horizon,
        feature_names=feature_columns(matrix),
        ticker=ticker.upper(),
        metrics=metrics,
    )

    return TrainResult(
        artifact=artifact,
        metrics=metrics,
        test_index=test_df.index,
        y_test=y_test,
        proba_test=proba if proba is not None else np.full(len(y_test), np.nan),
    )
