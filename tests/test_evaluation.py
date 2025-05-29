import pytest
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from mlops.evaluation.evaluation import (
    define_features_and_label,
    create_price_direction_label,
    prepare_features,
    split_data,
    train_linear_regression,
    train_logistic_regression
)

# Fixture that provides a realistic and balanced dataset for testing
@pytest.fixture
def sample_df():
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=10, freq='D'),
        'BTCUSDT_price': [100, 102, 101, 105, 107, 106, 108, 104, 110, 108],  # includes both increases and decreases
        'ETHUSDT_price': [50, 52, 53, 51, 55, 56, 54, 53, 58, 57],
        'BTCUSDT_funding_rate': [0.01, 0.02, 0.015, 0.017, 0.018, 0.019, 0.017, 0.016, 0.02, 0.019],
    }
    return pd.DataFrame(data)

# Test that checks the correct extraction of features and label column
def test_define_features_and_label(sample_df):
    symbols = ['ETHUSDT_price', 'BTCUSDT_funding_rate']
    features, label = define_features_and_label(sample_df, symbols)
    assert features == symbols
    assert label == "BTCUSDT_price"

# Test that ensures the price direction column is created and contains only 0 and 1
def test_create_price_direction_label(sample_df):
    df = create_price_direction_label(sample_df, 'BTCUSDT_price')
    assert 'price_direction' in df.columns
    assert set(df['price_direction'].unique()).issubset({0, 1})
    assert df.isnull().sum().sum() == 0  # check for no missing values

# Test that checks the output shapes and consistency of feature and label arrays
def test_prepare_features(sample_df):
    df = create_price_direction_label(sample_df, 'BTCUSDT_price')
    features, label = define_features_and_label(df, ['ETHUSDT_price', 'BTCUSDT_funding_rate'])
    X, y_reg, y_class = prepare_features(df, features, label)
    assert X.shape[0] == y_reg.shape[0] == y_class.shape[0]
    assert X.shape[1] == 2  # check number of features

# Test the data splitting logic and shape preservation
def test_split_data(sample_df):
    df = create_price_direction_label(sample_df, 'BTCUSDT_price')
    features, label = define_features_and_label(df, ['ETHUSDT_price', 'BTCUSDT_funding_rate'])
    X, y_reg, _ = prepare_features(df, features, label)
    X_train, X_test, y_train, y_test = split_data(X, y_reg, test_size=0.4)
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y_reg)

# Test training a linear regression model and ensure predictions are returned
def test_train_linear_regression(sample_df):
    df = create_price_direction_label(sample_df, 'BTCUSDT_price')
    features, label = define_features_and_label(df, ['ETHUSDT_price', 'BTCUSDT_funding_rate'])
    X, y_reg, _ = prepare_features(df, features, label)
    X_train, X_test, y_train, y_test = split_data(X, y_reg)
    model, preds = train_linear_regression(X_train, y_train, X_test, y_test)
    assert isinstance(model, LinearRegression)
    assert len(preds) == len(y_test)

# Test training a logistic regression model and ensure classification is working
def test_train_logistic_regression(sample_df):
    df = create_price_direction_label(sample_df, 'BTCUSDT_price')
    features, label = define_features_and_label(df, ['ETHUSDT_price', 'BTCUSDT_funding_rate'])
    X, _, y_class = prepare_features(df, features, label)
    X_train, X_test, y_train, y_test = split_data(X, y_class)

    # Check that we have at least two classes in the training set
    assert len(set(y_train)) >= 2

    model, preds = train_logistic_regression(X_train, y_train, X_test, y_test)
    assert isinstance(model, LogisticRegression)
    assert set(preds).issubset({0, 1})
