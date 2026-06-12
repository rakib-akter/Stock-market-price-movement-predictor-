"""Tests for feature engineering and the (critical) no-look-ahead property."""

from __future__ import annotations

import numpy as np

from backend.app.features.pipeline import (
    build_feature_matrix,
    feature_columns,
    make_labels,
)
from backend.app.features.returns import add_return_features
from backend.app.features.technical import add_technical_features, rsi


def test_rsi_within_bounds(ohlcv):
    values = rsi(ohlcv["adj_close"], 14).dropna()
    assert ((values >= 0) & (values <= 100)).all()


def test_return_features_present(ohlcv):
    out = add_return_features(ohlcv)
    for col in ["ret_1", "log_ret_1", "lag_ret_1", "mom_21"]:
        assert col in out.columns


def test_feature_matrix_has_no_nans(ohlcv, benchmark):
    matrix = build_feature_matrix(
        ohlcv, horizon=1, benchmarks={"SPY": benchmark}
    )
    assert len(matrix) > 0
    assert matrix.isna().sum().sum() == 0


def test_features_do_not_look_ahead(ohlcv):
    """Truncating the future must not change a feature value computed at time t.

    If any feature used future data, recomputing on a shorter series would change
    earlier rows. We assert the prefix is identical.
    """
    full = add_technical_features(add_return_features(ohlcv))
    cutoff = 400
    truncated = add_technical_features(add_return_features(ohlcv.iloc[:cutoff]))

    cols = feature_columns(full)
    # Compare a safely warmed-up window well inside both series.
    compare_cols = [c for c in cols if c in truncated.columns]
    a = full[compare_cols].iloc[250:cutoff].reset_index(drop=True)
    b = truncated[compare_cols].iloc[250:cutoff].reset_index(drop=True)
    assert np.allclose(a.fillna(0).values, b.fillna(0).values, atol=1e-9)


def test_labels_reference_the_future(ohlcv):
    labelled = make_labels(ohlcv, horizon=1)
    # y_dir_1 at row t must equal (close[t+1] > close[t]).
    close = ohlcv["adj_close"]
    expected = (close.shift(-1) > close).astype("Int64")
    assert (labelled["y_dir_1"].dropna() == expected.dropna()).all()
    # The last row's forward label is unknown (NaN) and gets dropped downstream.
    assert np.isnan(labelled["y_fwd_ret_1"].iloc[-1])
