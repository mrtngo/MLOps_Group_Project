import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import pytest

from mlops.data_validation.data_validation import load_config
from mlops.features.features import (
    create_price_direction_label,
    define_features_and_label,
    prepare_features,
)


@pytest.fixture
def sample_df():
    """Provides a DataFrame with timestamp, price, and sample features."""
    data = {
        "timestamp": pd.date_range(start="2024-01-01", periods=6, freq="D"),
        "BTCUSDT_price": [100, 101, 102, 100, 103, 99],
        "ETHUSDT_price": [50, 51, 52, 53, 54, 55],
        "BTCUSDT_funding_rate": [0.01, 0.02, 0.015, 0.017, 0.018, 0.019],
    }
    return pd.DataFrame(data)


def test_define_features_and_label_from_config(monkeypatch):
    """
    Test that the define_features_and_label function returns:
    - The correct feature columns based on a mocked config symbols list.
    - The expected label column name ("BTCUSDT_price").
    """
    mock_symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    mock_config = {"symbols": mock_symbols}
    monkeypatch.setattr(
        "mlops.data_validation.data_validation.load_config", lambda x=None: mock_config
    )
    expected_features = [
        f"{symbol}_price" for symbol in mock_symbols if symbol != "BTCUSDT"
    ] + [f"{symbol}_funding_rate" for symbol in mock_symbols]
    feature_cols, label_col = define_features_and_label(mock_config)
    assert set(feature_cols) == set(expected_features)
    assert label_col == "BTCUSDT_price"


def test_create_price_direction_label(sample_df):
    """
    Test that the create_price_direction_label function correctly adds a
    binary column 'price_direction' indicating whether the price increased
    from the previous row.

    This test ensures:
    - The new column is added.
    - It contains only 0s and 1s.
    - The DataFrame has no NaN values after the operation.

    This function is essential for transforming regression data into a
    classification problem by computing the direction of price movement.
    """
    df_result = create_price_direction_label(sample_df, "BTCUSDT_price")

    assert "price_direction" in df_result.columns
    assert df_result["price_direction"].isin([0, 1]).all()
    assert not df_result.isnull().values.any()


def test_prepare_features(sample_df):
    """
    Test that the prepare_features function returns:
    - A feature matrix X with the correct number of samples and columns.
    - A regression target (y_reg) matching the shape of X.
    - A classification target (y_class) containing only binary values.

    This function is critical because it prepares all the necessary inputs
    for training both regression and classification models. This test ensures
    that the function returns clean, aligned, valid data ready for modeling.
    """
    df = create_price_direction_label(sample_df, "BTCUSDT_price")
    feature_cols = ["ETHUSDT_price", "BTCUSDT_funding_rate"]
    label_col = "BTCUSDT_price"

    X, y_reg, y_class = prepare_features(df, feature_cols, label_col)

    assert X.shape[0] == y_reg.shape[0] == y_class.shape[0]
    assert X.shape[1] == len(feature_cols)
    assert not X.isnull().values.any()
    assert not y_reg.isnull().any()
    assert set(y_class.unique()).issubset({0, 1})


def test_define_features_and_label_empty_config():
    from src.mlops.features.features import define_features_and_label

    config = {}
    features, label = define_features_and_label(config)
    assert isinstance(features, list)


def test_create_price_direction_label_missing_label():
    import pandas as pd
    from src.mlops.features.features import create_price_direction_label

    df = pd.DataFrame({"timestamp": [1, 2, 3], "other": [1, 2, 3]})
    try:
        create_price_direction_label(df, "BTCUSDT_price")
    except Exception as e:
        assert isinstance(e, KeyError)


def test_prepare_features_missing_price_direction():
    import pandas as pd
    from src.mlops.features.features import prepare_features

    df = pd.DataFrame({"ETHUSDT_price": [1, 2], "BTCUSDT_price": [3, 4]})
    try:
        prepare_features(df, ["ETHUSDT_price"], "BTCUSDT_price")
    except Exception as e:
        assert isinstance(e, KeyError)


def test_select_features_missing_target():
    import pandas as pd
    from src.mlops.features.features import select_features

    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    try:
        select_features(df, ["a", "b"], "not_in_df", {})
    except Exception as e:
        assert isinstance(e, KeyError)


def test_get_training_and_testing_data_missing_file():
    from src.mlops.features.features import get_training_and_testing_data

    config = {"data_source": {"processed_path": "not_a_real_file.csv"}}
    train, test = get_training_and_testing_data(config)
    assert train is None and test is None




"""
Enhanced unit‑test suite for `mlops.features.features`.

These tests aim to raise line‑ and branch‑coverage by exercising:
 • Correlation‑based feature selection (both positive and negative correlations &
   varying thresholds).
 • Chronological train/test split logic with configurable `test_size`.
 • Price‑direction label creation for rising, falling and flat sequences.
 • Feature / target alignment after preparation.
 • Edge‑cases such as empty‑symbol configuration in `define_features_and_label`.

Run with:  `pytest -q tests/test_features_additional.py`
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from mlops.features.features import (
    create_price_direction_label,
    define_features_and_label,
    prepare_features,
    select_features,
    get_training_and_testing_data,
)


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_price_df() -> pd.DataFrame:
    """Generate a 10‑row price DataFrame with mixed movements."""
    rng = pd.date_range("2024-01-01", periods=10, freq="H")
    prices = pd.Series([100, 101, 99, 99, 102, 102, 103, 102, 101, 101])
    df = pd.DataFrame({
        "timestamp": rng,
        "BTCUSDT_price": prices,
        "ETHUSDT_price": prices * 0.05 + np.random.normal(0, 0.01, size=10),
        "BTCUSDT_funding_rate": np.random.uniform(-0.001, 0.001, size=10),
        "ETHUSDT_funding_rate": np.random.uniform(-0.001, 0.001, size=10),
    })
    return df


@pytest.fixture(scope="module")
def simple_config():
    return {
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "feature_engineering": {"feature_selection": {"correlation_threshold": 0.05}},
        "data_split": {"test_size": 0.3},
    }


# ---------------------------------------------------------------------------
# Tests for feature‑selection logic
# ---------------------------------------------------------------------------

def test_select_features_above_threshold(toy_price_df, simple_config):
    """Features with |corr| > threshold should be kept."""
    df = toy_price_df.copy()

    target = "BTCUSDT_price"
    # Exclude the target itself from candidate feature columns
    feature_cols = [c for c in df.columns if c.endswith("_price") and c != target]

    selected = select_features(df, feature_cols, target, simple_config)

    # ETHUSDT_price is highly correlated and should be retained
    assert "ETHUSDT_price" in selected
    # The target was never in `feature_cols`, so it should not appear in `selected`
    assert target not in selected



def test_select_features_below_threshold(toy_price_df, simple_config):
    """Lower the threshold so funding‑rate columns are now included."""
    cfg = simple_config.copy()
    cfg["feature_engineering"]["feature_selection"]["correlation_threshold"] = 0.0
    feature_cols = [c for c in toy_price_df.columns if c.endswith("rate")]
    selected = select_features(toy_price_df, feature_cols, "BTCUSDT_price", cfg)
    assert set(selected) == set(feature_cols)  # everything passes when threshold=0


# ---------------------------------------------------------------------------
# Tests for price‑direction label creation
# ---------------------------------------------------------------------------

def test_create_price_direction_label_values(toy_price_df):
    labeled = create_price_direction_label(toy_price_df, "BTCUSDT_price")
    # First row should be dropped (prev_price NaN), length = len(df) - 1
    assert len(labeled) == len(toy_price_df) - 1
    # Check explicit expected direction pattern
    expected = [1, 0, 0, 1, 0, 1, 0, 0, 0]
    assert labeled["price_direction"].tolist() == expected


# ---------------------------------------------------------------------------
# Tests for feature preparation and alignment
# ---------------------------------------------------------------------------

def test_prepare_features_alignment(toy_price_df):
    labeled = create_price_direction_label(toy_price_df, "BTCUSDT_price")
    X, y_reg, y_class = prepare_features(
        labeled,
        [c for c in labeled.columns if c.endswith("_price")],
        "BTCUSDT_price",
    )
    # Shapes
    assert len(X) == len(y_reg) == len(y_class)
    # No NaNs introduced
    assert not X.isnull().any().any()


# ---------------------------------------------------------------------------
# Tests for train/test split logic
# ---------------------------------------------------------------------------

def test_get_training_and_testing_data_split(simple_config, toy_price_df):
    train, test = get_training_and_testing_data(simple_config, toy_price_df)
    # Expected sizes (70/30 split rounded down)
    assert len(train) == 7 and len(test) == 3
    # Ensure chronological order is preserved
    assert train["timestamp"].max() < test["timestamp"].min()


# ---------------------------------------------------------------------------
# Edge‑case for feature / label definition
# ---------------------------------------------------------------------------

def test_define_features_and_label_empty():
    features, label = define_features_and_label({"symbols": []})
    assert features == [] and label == "BTCUSDT_price"
