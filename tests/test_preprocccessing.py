import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.preprocessing import StandardScaler
from mlops.preproccess import preproccessing as pp

from mlops.preproccess.preproccessing import (
    preprocess_pipeline,
    scale_features,
    scale_test_data,
    smote_oversample,
    split_data,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    df = pd.DataFrame({
        "feat1": np.random.rand(100),
        "feat2": np.random.rand(100),
        "feat3": np.random.rand(100),
        "feat4": np.random.rand(100),
        "BTCUSDT_price": np.random.rand(100) * 1000,
    })
    return df


def test_scale_features(sample_df):
    selected_cols = ["feat1", "feat2"]
    X_scaled, X_test_placeholder, scaler = scale_features(sample_df, selected_cols)

    assert X_scaled.shape == (100, 2)
    assert X_test_placeholder.size == 0
    assert isinstance(scaler, StandardScaler)

    np.testing.assert_almost_equal(X_scaled.mean(axis=0), np.zeros(2), decimal=1)
    np.testing.assert_almost_equal(X_scaled.std(axis=0), np.ones(2), decimal=1)


def test_scale_test_data(sample_df):
    selected_cols = ["feat1", "feat2"]
    X_train = sample_df[selected_cols].iloc[:80]
    X_test = sample_df[selected_cols].iloc[80:]

    _, _, scaler = scale_features(X_train, selected_cols)
    X_test_scaled = scale_test_data(X_test, scaler, selected_cols)

    assert X_test_scaled.shape == (20, 2)
    assert isinstance(X_test_scaled, np.ndarray)


def test_split_data(sample_df):
    X = sample_df[["feat1", "feat2"]]
    y = sample_df["feat3"]
    dummy_config = {}
    X_train, X_test, y_train, y_test = split_data(X, y, config=dummy_config)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert X_train.shape[1] == X.shape[1]


def test_split_data_custom_config(sample_df):
    X = sample_df[["feat1", "feat2"]]
    y = sample_df["feat3"]
    config = {"data_split": {"test_size": 0.3, "random_state": 123}}
    X_train, X_test, _, _ = split_data(X, y, config)
    assert len(X_test) == 30


def test_smote_oversample_applies():
    X = np.random.rand(30, 2)
    y = [0] * 24 + [1] * 6
    config = {"preprocessing": {"sampling": {"threshold_ratio": 2, "params": {"random_state": 1}}}}
    X_res, y_res = smote_oversample(X, y, config=config)
    assert len(X_res) > len(X)
    assert len(X_res) == len(y_res)
    unique, counts = np.unique(y_res, return_counts=True)
    assert set(unique) == {0, 1}
    assert abs(counts[0] - counts[1]) <= 1


def test_smote_oversample_skips():
    X = np.random.rand(20, 2)
    y = [0] * 10 + [1] * 10
    dummy_config = {}
    X_res, y_res = smote_oversample(X, y, config=dummy_config)
    assert len(X_res) == len(X)
    assert np.allclose(X, X_res)
    assert np.array_equal(y, y_res)


def test_smote_invalid_input():
    X = pd.DataFrame({"a": ["bad", "data", "test"]})  # Non-numeric input
    y = [0, 1, 0]
    with pytest.raises(ValueError):
        smote_oversample(X, y, config={})

def test_smote_oversample_one_class():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 1, 1])
    config = {}
    X_res, y_res = smote_oversample(X, y, config)
    assert (X == X_res).all() and (y == y_res).all()


def test_scale_features_non_numeric():
    df = pd.DataFrame({"a": ["x", "y"]})
    with pytest.raises(ValueError):
        scale_features(df, ["a"])


def test_scale_test_data_unfit_scaler():
    df = pd.DataFrame({"a": [1, 2]})
    scaler = StandardScaler()
    with pytest.raises(Exception):
        scale_test_data(df, scaler, ["a"])


def test_preprocess_pipeline(sample_df):
    feature_cols = ["feat1", "feat2"]
    X = sample_df[feature_cols]
    y = [0] * 80 + [1] * 20
    X_train, X_test, y_train, _ = split_data(X, y, config={})
    pipeline_result = preprocess_pipeline(
        X_train, X_test, y_train, feature_cols, apply_smote=True, config={}
    )
    X_train_prep, X_test_prep, y_train_prep, y_train_orig, scaler = pipeline_result
    assert X_train_prep.shape[1] == len(feature_cols)
    assert X_test_prep.shape == X_test.shape
    assert len(y_train_prep) == len(X_train_prep)
    assert isinstance(scaler, StandardScaler)


def test_preprocess_pipeline_no_smote(sample_df):
    feature_cols = ["feat1", "feat2"]
    X = sample_df[feature_cols]
    y = [0, 1] * 50
    X_train, X_test, y_train, _ = split_data(X, y, config={})
    result = preprocess_pipeline(X_train, X_test, y_train, feature_cols, config={}, apply_smote=False)
    X_train_prep, X_test_prep, y_train_prep, y_train_orig, scaler = result
    assert X_train_prep.shape == (80, 2)
    assert X_test_prep.shape == (20, 2)
    assert list(y_train_prep) == y_train


def test_preprocess_pipeline_smote_one_class():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([1, 1, 1])
    config = {}
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(
        X, X, y, ["a"], config, apply_smote=True
    )
    assert (y_train == y).all()


def test_split_data_empty():
    config = {}
    X = pd.DataFrame()
    y = pd.Series(dtype=float)
    with pytest.raises(ValueError):
        split_data(X, y, config)


def test_main_logger_manual():
    with patch("mlops.preproccess.preproccessing.logger.info") as mock_log:
        pp.logger.info("preproccessing.py - Use functions by importing them in other modules")
        mock_log.assert_called_with("preproccessing.py - Use functions by importing them in other modules")