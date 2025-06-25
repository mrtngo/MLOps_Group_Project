# src/feature_engineering/feature_eng.py
"""
Feature engineering for crypto prediction (adapted from teacher's pattern).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CryptoRiskScore(BaseEstimator, TransformerMixin):
    """
    Creates a crypto risk score based on price volatility and funding rates.

    Similar to teacher's RiskScore but for crypto data.
    """

    def __init__(self, price_columns, funding_columns):
        self.price_columns = price_columns
        self.funding_columns = funding_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        # Ensure all price columns exist
        for col in self.price_columns:
            if col not in X.columns:
                X[col] = 0

        # Ensure all funding rate columns exist
        for col in self.funding_columns:
            if col not in X.columns:
                X[col] = 0

        # Calculate price volatility risk
        price_data = (
            X[self.price_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        )
        price_volatility = price_data.std(axis=1)

        # Calculate funding rate risk (absolute values)
        funding_data = (
            X[self.funding_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        )
        funding_risk = funding_data.abs().sum(axis=1)

        # Combined risk score
        X["crypto_risk_score"] = price_volatility + funding_risk
        return X


class PriceMovementFeatures(BaseEstimator, TransformerMixin):
    """
    Creates price movement features for crypto prediction.
    """

    def __init__(self, price_columns):
        self.price_columns = price_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        for col in self.price_columns:
            if col in X.columns:
                # Price change percentage
                X[f"{col}_pct_change"] = X[col].pct_change().fillna(0)

                # Moving average (simple 3-period)
                X[f"{col}_ma3"] = X[col].rolling(window=3, min_periods=1).mean()

                # Price momentum
                X[f"{col}_momentum"] = X[col] - X[f"{col}_ma3"]

        return X


FEATURE_TRANSFORMERS = {
    "crypto_risk_score": lambda config: CryptoRiskScore(
        price_columns=[
            col for col in config.get("raw_features", []) if "_price" in col
        ],
        funding_columns=[
            col for col in config.get("raw_features", []) if "_funding_rate" in col
        ],
    ),
    "price_movement": lambda config: PriceMovementFeatures(
        price_columns=[col for col in config.get("raw_features", []) if "_price" in col]
    ),
}
