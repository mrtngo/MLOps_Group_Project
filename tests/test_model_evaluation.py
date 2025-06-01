import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from mlops.evaluation.evaluation import ModelEvaluator

# === DummyScaler class for mocking ===
class DummyScaler:
    def transform(self, X):
        return X.to_numpy()
    # Add feature names to avoid sklearn warnings/errors
    feature_names_in_ = ["ETHUSDT_price", "BTCUSDT_funding_rate"]

# === Fixtures ===
@pytest.fixture
def dummy_dataframe():
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
        "BTCUSDT_price": [100, 101, 102, 100, 103, 99, 104, 106, 110, 108],
        "ETHUSDT_price": [50, 51, 52, 53, 54, 55, 56, 57, 58, 57],
        "BTCUSDT_funding_rate": [0.01, 0.02, 0.015, 0.017, 0.018, 0.019, 0.02, 0.021, 0.022, 0.019]
    })

@pytest.fixture
def dummy_pipeline():
    return {
        "scaler": DummyScaler(),
        "selected_features_reg": ["ETHUSDT_price"],
        "selected_features_class": ["BTCUSDT_funding_rate"],
        "all_feature_cols": ["ETHUSDT_price", "BTCUSDT_funding_rate"]
    }

# === Test ===
def test_prepare_test_data(dummy_dataframe, dummy_pipeline):
    evaluator = ModelEvaluator()
    evaluator.preprocessing_pipeline = dummy_pipeline

    with patch("mlops.evaluation.evaluation.define_features_and_label") as mock_define:
        mock_define.return_value = (
            ["ETHUSDT_price", "BTCUSDT_funding_rate"],
            "BTCUSDT_price"
        )
        X_reg, X_class, y_reg, y_class = evaluator.prepare_test_data(dummy_dataframe)

    assert X_reg.shape[1] == 1
    assert X_class.shape[1] == 1
    assert len(y_reg) > 0
    assert len(y_class) > 0
