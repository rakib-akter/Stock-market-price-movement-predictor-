"""Feature engineering: quant-style features and labels.

Every feature uses only information available at or before bar ``t``. Only the
label functions in :mod:`backend.app.features.pipeline` look forward.
"""

from backend.app.features.market import add_market_features
from backend.app.features.pipeline import build_feature_matrix, make_labels
from backend.app.features.returns import add_return_features
from backend.app.features.technical import add_technical_features
from backend.app.features.volatility import add_volatility_features

__all__ = [
    "add_return_features",
    "add_technical_features",
    "add_volatility_features",
    "add_market_features",
    "make_labels",
    "build_feature_matrix",
]
