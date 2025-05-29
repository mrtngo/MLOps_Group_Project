import pytest
import pandas as pd
from mlops.features import features as features_module  # Avoid conflict with 'features' variable name

# Fixture: provides a small DataFrame with timestamp, price, and sample features
@pytest.fixture
def sample_df():
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=6, freq='D'),
        'BTCUSDT_price': [100, 101, 102, 100, 103, 99],
        'ETHUSDT_price': [50, 51, 52, 53, 54, 55],
        'BTCUSDT_funding_rate': [0.01, 0.02, 0.015, 0.017, 0.018, 0.019]
    }
    return pd.DataFrame(data)

def test_define_features_and_label(sample_df):
    """
    Test that the define_features_and_label function returns:
    - The same list of features as provided.
    - The expected label column name ("BTCUSDT_price").

    This test ensures that the function correctly identifies which columns will be used as
    inputs (features) and which will be the target for prediction (label), which is a
    critical setup step for model training.
    """
    symbols = ['ETHUSDT_price', 'BTCUSDT_funding_rate']
    feature_cols, label_col = features_module.define_features_and_label(sample_df, symbols)

    assert feature_cols == symbols
    assert label_col == "BTCUSDT_price"

def test_create_price_direction_label(sample_df):
    """
    Test that the create_price_direction_label function correctly adds a binary column
    'price_direction' indicating whether the price increased from the previous row.

    This test ensures:
    - The new column is added.
    - It contains only 0s and 1s.
    - The DataFrame has no NaN values after the operation.

    This function is essential for transforming regression data into a classification
    problem by computing the direction of price movement.
    """
    df_result = features_module.create_price_direction_label(sample_df, 'BTCUSDT_price')

    assert 'price_direction' in df_result.columns
    assert df_result['price_direction'].isin([0, 1]).all()
    assert not df_result.isnull().values.any()

def test_prepare_features(sample_df):
    """
    Test that the prepare_features function returns:
    - A feature matrix X with the correct number of samples and columns.
    - A regression target (y_reg) matching the shape of X.
    - A classification target (y_class) containing only binary values.

    This function is critical because it prepares all the necessary inputs for training
    both regression and classification models. This test ensures that the function
    returns clean, aligned, and valid data ready for modeling.
    """
    df = features_module.create_price_direction_label(sample_df, 'BTCUSDT_price')
    feature_cols = ['ETHUSDT_price', 'BTCUSDT_funding_rate']
    label_col = 'BTCUSDT_price'
    
    X, y_reg, y_class = features_module.prepare_features(df, feature_cols, label_col)

    assert X.shape[0] == y_reg.shape[0] == y_class.shape[0]
    assert X.shape[1] == len(feature_cols)
    assert not X.isnull().values.any()
    assert not y_reg.isnull().any()
    assert set(y_class.unique()).issubset({0, 1})
