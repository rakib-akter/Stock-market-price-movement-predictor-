# Notebooks

Exploratory research lives here. Notebooks are for *investigation*; once an idea
proves out, promote it into a tested module under `backend/app/`.

## Suggested notebooks

| Notebook | Purpose |
| --- | --- |
| `01_data_exploration` | Pull a few tickers, inspect gaps, splits, adjusted vs. raw close. |
| `02_feature_analysis` | Distributions, correlations, feature/target mutual information. |
| `03_model_comparison` | LR vs RF vs GBM vs XGB on walk-forward AUC. |
| `04_backtest_deepdive` | Equity curves, drawdowns, cost sensitivity, sub-period stability. |

## Discipline

- **Never** compute statistics on the full series before splitting (leakage).
- Re-import from `backend.app` rather than copy-pasting logic, so the notebook and
  the production pipeline cannot drift apart.
- Treat every promising backtest as a hypothesis to be re-validated out-of-sample.

See [`../docs/warnings.md`](../docs/warnings.md).
