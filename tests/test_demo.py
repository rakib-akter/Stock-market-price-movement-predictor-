"""Tests for offline demo mode (synthetic data + end-to-end service path)."""

from __future__ import annotations

from backend.app.data.demo import generate_demo_ohlcv
from backend.app.data.loader import load_price_panel
from backend.app.service import build_features_for, predict_for


def test_demo_data_is_deterministic_per_ticker():
    a1 = generate_demo_ohlcv("AAPL", start="2020-01-01", end="2022-01-01")
    a2 = generate_demo_ohlcv("AAPL", start="2020-01-01", end="2022-01-01")
    b = generate_demo_ohlcv("MSFT", start="2020-01-01", end="2022-01-01")
    # Same ticker → identical series; different ticker → different series.
    assert a1["close"].equals(a2["close"])
    assert not a1["close"].equals(b["close"].reindex(a1.index))


def test_demo_data_obeys_ohlcv_contract():
    df = generate_demo_ohlcv("NVDA", start="2021-01-01", end="2022-01-01")
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert df.index.name == "date"
    # High is the max and low is the min of the bar.
    assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (df[["open", "high", "low", "close"]] > 0).all().all()


def test_loader_demo_flag_forces_synthetic():
    df = load_price_panel("TSLA", start="2021-01-01", end="2022-01-01", demo=True)
    assert len(df) > 0
    assert df["adj_close"].notna().all()


def test_service_runs_end_to_end_in_demo_mode():
    # demo_mode defaults to True, so this works with no network.
    bundle = build_features_for("AAPL", horizon=1)
    assert len(bundle.matrix) > 0
    pred = predict_for("AAPL", model_name="logistic", horizon=1)
    assert pred["direction"] in {"UP", "DOWN"}
    assert 0.0 <= pred["confidence"] <= 1.0
