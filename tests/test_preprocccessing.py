import pandas as pd
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from src.mlops.preproccess.preproccessing import select_features, scale_features, smote_oversample

@pytest.fixture
def sample_df():
    np.random.seed(42)
    df = pd.DataFrame({
        'feat1': np.random.rand(100),
        'feat2': np.random.rand(100),
        'feat3': np.random.rand(100),
        'feat4': np.random.rand(100),
        'BTCUSDT_price': np.random.rand(100) * 1000
    })
    return df

def test_select_features(sample_df):
    selected = select_features(sample_df, ['feat1', 'feat2', 'feat3', 'feat4'], 'BTCUSDT_price', top_n=2)
    assert isinstance(selected, list)
    assert len(selected) == 2
    for col in selected:
        assert col in ['feat1', 'feat2', 'feat3', 'feat4']

def test_scale_features(sample_df):
    selected_cols = ['feat1', 'feat2']
    X_scaled, scaler = scale_features(sample_df, selected_cols)
    assert X_scaled.shape == (100, 2)
    assert isinstance(scaler, StandardScaler)
    # Check mean ~ 0 and std ~ 1
    np.testing.assert_almost_equal(X_scaled.mean(axis=0), np.array([0, 0]), decimal=1)

def test_smote_oversample_applies():
    X = np.random.rand(30, 2)
    y = [0]*24 + [1]*6  # יחס 4:1, אך עם 6 דגימות במחלקת המיעוט
    X_res, y_res = smote_oversample(X, y)
    assert len(X_res) > len(X)
    assert len(X_res) == len(y_res)
    unique, counts = np.unique(y_res, return_counts=True)
    ratio = max(counts) / min(counts)
    assert ratio <= 1.5


def test_smote_oversample_skips():
    X = np.random.rand(20, 2)
    y = [0]*10 + [1]*10  # Balanced
    X_res, y_res = smote_oversample(X, y)
    assert len(X_res) == len(X)
    assert np.array_equal(X, X_res)
    assert np.array_equal(y, y_res)
