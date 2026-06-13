"""Streamlit research dashboard.

Run with::

    streamlit run frontend/streamlit_app.py

The dashboard talks directly to the backend service layer (no API needed), so it
works stand-alone. Set ``USE_API=1`` to route through the FastAPI service instead.

Sections:
    * ticker selector + date range
    * price chart with indicators (SMA, Bollinger Bands)
    * model prediction + confidence
    * walk-forward backtest: equity curve vs. buy-and-hold + metrics
    * feature importance
    * recent signals table
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backend.app.backtesting.engine import BacktestConfig
from backend.app.backtesting.walkforward import walk_forward_backtest
from backend.app.config import settings
from backend.app.features.pipeline import feature_columns
from backend.app.models.registry import available_models
from backend.app.models.train import train_classifier
from backend.app.service import build_features_for

st.set_page_config(page_title="Quant Movement Predictor", layout="wide")


# --------------------------------------------------------------------------- #
# Caching wrappers (Streamlit reruns the whole script on every interaction).
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Building features…", ttl=3600)
def _features(ticker: str, horizon: int) -> pd.DataFrame:
    return build_features_for(ticker, horizon=horizon).matrix


@st.cache_data(show_spinner="Training model…", ttl=3600)
def _train(ticker: str, model_name: str, horizon: int, calibrate: bool) -> dict:
    matrix = _features(ticker, horizon)
    result = train_classifier(
        matrix, ticker=ticker, model_name=model_name, horizon=horizon, calibrate=calibrate
    )
    # Feature importance comes from an uncalibrated fit (calibration wrappers hide
    # coef_/feature_importances_), so train a plain one just for the chart.
    plain = train_classifier(matrix, ticker=ticker, model_name=model_name, horizon=horizon)
    importances = _feature_importance(plain.artifact.estimator, feature_columns(matrix))
    return {"metrics": result.metrics, "importances": importances, "calibrated": calibrate}


@st.cache_data(show_spinner="Running walk-forward backtest…", ttl=3600)
def _backtest(ticker: str, model_name: str, horizon: int, n_splits: int,
              allow_short: bool, threshold: float, sizing: str) -> dict:
    matrix = _features(ticker, horizon)
    bt = walk_forward_backtest(
        matrix, model_name=model_name, horizon=horizon, n_splits=n_splits,
        config=BacktestConfig(
            allow_short=allow_short, confidence_threshold=threshold, sizing=sizing
        ),
    )
    return {
        "metrics": bt.metrics,
        "equity": bt.equity,
        "benchmark": bt.benchmark_equity,
        "signals": bt.signals,
    }


def _feature_importance(estimator, names: list[str]) -> pd.Series:
    """Extract feature importance (tree models) or |coef| (linear)."""
    model = estimator
    if hasattr(estimator, "named_steps"):  # sklearn Pipeline
        model = estimator.named_steps.get("clf", estimator)
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coef_"):
        vals = np.abs(np.ravel(model.coef_))
    else:
        return pd.Series(dtype=float)
    return pd.Series(vals, index=names).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Controls")
ticker = st.sidebar.selectbox("Ticker", settings.default_tickers, index=0)
custom = st.sidebar.text_input("…or enter a symbol").strip().upper()
if custom:
    ticker = custom

model_name = st.sidebar.selectbox("Model", available_models(), index=0)
horizon = st.sidebar.slider("Prediction horizon (days)", 1, 21, 1)
n_splits = st.sidebar.slider("Walk-forward folds", 2, 12, 5)
allow_short = st.sidebar.checkbox("Allow short positions", value=False)
threshold = st.sidebar.slider("Confidence threshold", 0.0, 0.9, 0.0, 0.05)
sizing = st.sidebar.selectbox(
    "Position sizing", ["binary", "confidence", "vol_target"], index=0,
    help="binary: full position · confidence: scale by |edge| · vol_target: target annual vol",
)
calibrate = st.sidebar.checkbox(
    "Calibrate probabilities", value=False,
    help="Time-series-safe calibration so the confidence score is a real probability.",
)

st.sidebar.markdown("---")
st.sidebar.warning("Research tool — not financial advice. See docs/warnings.md.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("📈 Quant Stock-Movement Predictor")
st.caption(
    f"Target: **next-{horizon}-day direction** · Model: **{model_name}** · "
    f"Costs: {settings.transaction_cost_bps + settings.slippage_bps:.0f} bps/trade"
)

try:
    matrix = _features(ticker, horizon)
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load data for {ticker}: {exc}")
    st.stop()


# --------------------------------------------------------------------------- #
# Price chart with indicators
# --------------------------------------------------------------------------- #
st.subheader(f"{ticker} — price & indicators")
fig = go.Figure()
fig.add_trace(go.Scatter(x=matrix.index, y=matrix["adj_close"], name="Adj Close"))
for col, dash in [("sma_20", "dot"), ("sma_50", "dash")]:
    if col in matrix:
        fig.add_trace(go.Scatter(x=matrix.index, y=matrix[col], name=col.upper(),
                                 line=dict(dash=dash)))
if {"bb_upper_20", "bb_lower_20"} <= set(matrix.columns):
    fig.add_trace(go.Scatter(x=matrix.index, y=matrix["bb_upper_20"], name="BB upper",
                             line=dict(width=0.5, color="gray")))
    fig.add_trace(go.Scatter(x=matrix.index, y=matrix["bb_lower_20"], name="BB lower",
                             line=dict(width=0.5, color="gray"), fill="tonexty",
                             fillcolor="rgba(128,128,128,0.1)"))
fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Prediction + training metrics
# --------------------------------------------------------------------------- #
train_out = _train(ticker, model_name, horizon, calibrate)
metrics = train_out["metrics"]
if calibrate:
    st.caption("✅ Probabilities calibrated (time-series-safe). Confidence ≈ real probability.")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Hold-out accuracy", f"{metrics['accuracy']:.1%}",
            f"{metrics['accuracy'] - metrics['baseline_accuracy']:+.1%} vs baseline")
col2.metric("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}")
col3.metric("Brier", f"{metrics.get('brier', float('nan')):.3f}")
col4.metric("Precision", f"{metrics['precision']:.1%}")
col5.metric("Test samples", f"{metrics['n_test']}")


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
st.subheader("Walk-forward backtest vs. buy-and-hold")
bt = _backtest(ticker, model_name, horizon, n_splits, allow_short, threshold, sizing)
m = bt["metrics"]
st.caption(f"Sizing: **{sizing}** · avg exposure {m.get('avg_exposure', 0):.2f}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total return", f"{m['total_return']:.1%}",
          f"{m.get('excess_total_return', 0):+.1%} vs B&H")
c2.metric("Sharpe", f"{m['sharpe']:.2f}")
c3.metric("Max drawdown", f"{m['max_drawdown']:.1%}")
c4.metric("Win rate", f"{m['win_rate']:.1%}")
c5.metric("# Trades", f"{m.get('n_trades', 0)}")

eq_fig = go.Figure()
eq_fig.add_trace(go.Scatter(x=bt["equity"].index, y=bt["equity"], name="Strategy"))
eq_fig.add_trace(go.Scatter(x=bt["benchmark"].index, y=bt["benchmark"],
                            name="Buy & Hold", line=dict(dash="dash")))
eq_fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                     yaxis_title="Equity")
st.plotly_chart(eq_fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Feature importance + recent signals
# --------------------------------------------------------------------------- #
left, right = st.columns(2)

with left:
    st.subheader("Feature importance")
    importances = train_out["importances"]
    if len(importances):
        top = importances.head(15).iloc[::-1]
        imp_fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h"))
        imp_fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(imp_fig, use_container_width=True)
    else:
        st.info("This model does not expose feature importances.")

with right:
    st.subheader("Recent signals")
    sig = bt["signals"].tail(15).copy()
    sig["direction"] = np.where(sig["position"] > 0, "LONG",
                                np.where(sig["position"] < 0, "SHORT", "FLAT"))
    show = sig[["prob_up", "direction", "net_return"]].copy()
    show["prob_up"] = (show["prob_up"] * 100).round(1)
    show["net_return"] = (show["net_return"] * 100).round(2)
    st.dataframe(show.rename(columns={"prob_up": "P(up) %", "net_return": "Net ret %"}),
                 use_container_width=True)

st.caption("Past performance — especially backtested — does not predict future results.")
