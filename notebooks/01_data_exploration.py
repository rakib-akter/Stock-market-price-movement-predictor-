"""01 — Data exploration (runnable as a script or in a Jupyter `# %%` session).

Open in VS Code / Jupyter and run cell-by-cell, or `python notebooks/01_data_exploration.py`.
This is a *research scratchpad*, not production code.
"""

# %% Imports
import matplotlib.pyplot as plt  # noqa: E402  (notebooks install matplotlib ad hoc)

from backend.app.data.loader import load_price_panel
from backend.app.features.pipeline import build_feature_matrix

# %% Load one ticker
TICKER = "AAPL"
prices = load_price_panel(TICKER, start="2018-01-01")
print(prices.tail())
print("\nMissing values per column:\n", prices.isna().sum())

# %% Plot adjusted close
prices["adj_close"].plot(title=f"{TICKER} adjusted close", figsize=(11, 4))
plt.tight_layout()
plt.show()

# %% Build features and inspect the target balance
matrix = build_feature_matrix(prices, horizon=1)
print("Feature matrix shape:", matrix.shape)
print("\nUp/down balance (y_dir_1):\n", matrix["y_dir_1"].value_counts(normalize=True))

# %% Feature correlation with the next-day return (quick signal scan)
corr = (
    matrix.drop(columns=[c for c in matrix.columns if c.startswith("y_")])
    .corrwith(matrix["y_fwd_ret_1"])
    .sort_values(key=abs, ascending=False)
)
print("\nTop correlations with next-day return:\n", corr.head(12))
