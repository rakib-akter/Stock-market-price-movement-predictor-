"""Walk-forward model comparison.

Compares several models head-to-head on the *same* out-of-sample walk-forward
predictions, so the ranking is honest and apples-to-apples. This is how you pick a
model — never by in-sample fit, and never on the final test set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

from backend.app.backtesting.engine import BacktestConfig, backtest_signals
from backend.app.backtesting.walkforward import walk_forward_predictions
from backend.app.models.registry import available_models
from backend.app.utils.logging import get_logger

logger = get_logger(__name__)


def compare_models(
    matrix: pd.DataFrame,
    model_names: list[str] | None = None,
    horizon: int = 1,
    n_splits: int = 5,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """Rank models by walk-forward OOS classification and backtest metrics.

    Returns a DataFrame indexed by model name, sorted by ROC-AUC descending. Models
    whose optional dependency is missing (xgboost/lightgbm) are skipped with a log.
    """
    model_names = model_names or available_models()
    target = f"y_dir_{horizon}"
    y_true = matrix[target].astype(int)
    config = config or BacktestConfig()

    rows: list[dict] = []
    for name in model_names:
        try:
            prob = walk_forward_predictions(
                matrix, model_name=name, horizon=horizon, n_splits=n_splits
            )
        except ImportError as exc:  # optional dep not installed
            logger.warning("Skipping %s: %s", name, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model %s failed: %s", name, exc)
            continue

        y = y_true.loc[prob.index]
        pred = (prob >= 0.5).astype(int)

        row = {
            "model": name,
            "n_oos": int(len(prob)),
            "accuracy": float(accuracy_score(y, pred)),
            "brier": float(brier_score_loss(y, prob)),
        }
        if len(np.unique(y)) == 2:
            row["roc_auc"] = float(roc_auc_score(y, prob))

        bt = backtest_signals(matrix.loc[prob.index, ["adj_close"]], prob, config)
        row["bt_total_return"] = bt.metrics["total_return"]
        row["bt_sharpe"] = bt.metrics["sharpe"]
        row["bt_max_drawdown"] = bt.metrics["max_drawdown"]
        rows.append(row)

    if not rows:
        raise RuntimeError("No models could be evaluated.")

    df = pd.DataFrame(rows).set_index("model")
    sort_col = "roc_auc" if "roc_auc" in df.columns else "accuracy"
    return df.sort_values(sort_col, ascending=False)
