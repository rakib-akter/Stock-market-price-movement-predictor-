"""Time-aware splitting utilities.

These functions are the project's main defense against look-ahead bias and data
leakage. **Never** use ``sklearn.model_selection.train_test_split`` (which
shuffles) on this data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backend.app.config import settings


@dataclass(frozen=True)
class Fold:
    """A single walk-forward fold expressed as integer row positions."""

    train_idx: pd.Index
    test_idx: pd.Index


def time_train_test_split(
    df: pd.DataFrame, test_size: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered frame into earlier ``train`` and later ``test``.

    The split point is chosen by position, so the test set is always strictly
    *after* the train set in time.
    """
    test_size = settings.test_size if test_size is None else test_size
    if not 0 < test_size < 1:
        raise ValueError("test_size must be in (0, 1)")

    df = df.sort_index()
    cutoff = int(len(df) * (1 - test_size))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def walk_forward_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_min_size: int | None = None,
    embargo: int = 0,
    expanding: bool = True,
) -> list[Fold]:
    """Generate walk-forward folds over a time-ordered index.

    Parameters
    ----------
    n_splits:
        Number of sequential test windows.
    train_min_size:
        Minimum number of rows in the first training window. Defaults to one
        test-window length.
    embargo:
        Number of rows to *skip* between the end of train and the start of test.
        This "purges" leakage from features/labels that span the boundary
        (López de Prado's embargo). With a horizon-``h`` label, set ``embargo>=h``.
    expanding:
        If ``True``, training windows grow (anchored start). If ``False``, use a
        rolling fixed-size window.
    """
    df = df.sort_index()
    n = len(df)
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1")

    test_size = n // (n_splits + 1)
    if test_size == 0:
        raise ValueError("Not enough rows for the requested number of splits.")

    train_min_size = train_min_size or test_size
    folds: list[Fold] = []

    for k in range(n_splits):
        test_start = train_min_size + k * test_size + embargo
        test_end = test_start + test_size
        if test_start >= n:
            break
        test_end = min(test_end, n)

        train_start = 0 if expanding else max(0, test_start - embargo - train_min_size)
        train_end = test_start - embargo

        if train_end - train_start < train_min_size:
            continue

        folds.append(
            Fold(
                train_idx=df.index[train_start:train_end],
                test_idx=df.index[test_start:test_end],
            )
        )

    return folds
