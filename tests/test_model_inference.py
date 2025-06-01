import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from mlops.inference.inference import ModelInferencer

# --- Dummy components ---
class DummyModel:
    def predict(self, X):
        return np.ones(len(X))

    def predict_proba(self, X):
        return np.array([[0.3, 0.7]] * len(X))

class DummyScaler:
    def transform(self, X):
        return X.to_numpy()
    feature_names_in_ = ["ETHUSDT_price", "BTCUSDT_funding_rate"]

# --- Fixtures ---
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
        "BTCUSDT_price": [100, 101, 102, 103, 104],
        "ETHUSDT_price": [50, 51, 52, 53, 54],
        "BTCUSDT_funding_rate": [0.01, 0.02, 0.015, 0.017, 0.019]
    })

@pytest.fixture
def dummy_pipeline():
    return {
        "scaler": DummyScaler(),
        "selected_features_reg": ["ETHUSDT_price"],
        "selected_features_class": ["BTCUSDT_funding_rate"],
        "all_feature_cols": ["ETHUSDT_price", "BTCUSDT_funding_rate"]
    }

# --- Tests ---
@patch("mlops.inference.inference.define_features_and_label")
def test_predict_price(mock_define, sample_df, dummy_pipeline):
    mock_define.return_value = (
        ["ETHUSDT_price", "BTCUSDT_funding_rate"],
        "BTCUSDT_price"
    )
    
    inferencer = ModelInferencer()
    inferencer.price_model = DummyModel()
    inferencer.preprocessing_pipeline = dummy_pipeline
    
    predictions = inferencer.predict_price(sample_df)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(sample_df)

@patch("mlops.inference.inference.define_features_and_label")
def test_predict_direction(mock_define, sample_df, dummy_pipeline):
    mock_define.return_value = (
        ["ETHUSDT_price", "BTCUSDT_funding_rate"],
        "BTCUSDT_price"
    )

    inferencer = ModelInferencer()
    inferencer.direction_model = DummyModel()
    inferencer.preprocessing_pipeline = dummy_pipeline

    directions, probs = inferencer.predict_direction(sample_df)
    assert isinstance(directions, np.ndarray)
    assert isinstance(probs, np.ndarray)
    assert len(directions) == len(sample_df)
    assert len(probs) == len(sample_df)

@patch("mlops.inference.inference.define_features_and_label")
def test_predict_both(mock_define, sample_df, dummy_pipeline):
    mock_define.return_value = (
        ["ETHUSDT_price", "BTCUSDT_funding_rate"],
        "BTCUSDT_price"
    )

    inferencer = ModelInferencer()
    inferencer.price_model = DummyModel()
    inferencer.direction_model = DummyModel()
    inferencer.preprocessing_pipeline = dummy_pipeline

    results = inferencer.predict_both(sample_df)
    assert "price_predictions" in results
    assert "direction_predictions" in results
    assert "direction_probabilities" in results
    assert len(results["price_predictions"]) == len(sample_df)
