"""Tests for cleaning and time-aware splitting."""

from __future__ import annotations

import numpy as np

from backend.app.data.clean import clean_ohlcv
from backend.app.data.splits import time_train_test_split, walk_forward_folds


def test_clean_handles_missing_and_keeps_positive(ohlcv):
    dirty = ohlcv.copy()
    dirty.iloc[5, dirty.columns.get_loc("volume")] = np.nan
    cleaned = clean_ohlcv(dirty)
    # No NaNs in price columns, all prices positive.
    assert cleaned[["open", "high", "low", "close"]].isna().sum().sum() == 0
    assert (cleaned[["open", "high", "low", "close"]] > 0).all().all()


def test_time_split_is_chronological(ohlcv):
    train, test = time_train_test_split(ohlcv, test_size=0.2)
    assert len(train) + len(test) == len(ohlcv)
    # Every train timestamp precedes every test timestamp.
    assert train.index.max() < test.index.min()
    assert np.isclose(len(test) / len(ohlcv), 0.2, atol=0.01)


def test_walk_forward_folds_are_ordered_and_non_overlapping(ohlcv):
    folds = walk_forward_folds(ohlcv, n_splits=4, embargo=1)
    assert len(folds) >= 1
    for fold in folds:
        # Train strictly precedes test (embargo enforces the gap).
        assert fold.train_idx.max() < fold.test_idx.min()
    # Test windows advance through time.
    starts = [f.test_idx.min() for f in folds]
    assert starts == sorted(starts)


def test_embargo_creates_gap(ohlcv):
    embargo = 5
    folds = walk_forward_folds(ohlcv, n_splits=3, embargo=embargo)
    for fold in folds:
        train_end_pos = ohlcv.index.get_loc(fold.train_idx.max())
        test_start_pos = ohlcv.index.get_loc(fold.test_idx.min())
        assert test_start_pos - train_end_pos >= embargo
