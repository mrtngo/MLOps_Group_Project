
import pytest
import pandas as pd
import logging
from io import StringIO
from data_validation import (
    validate_data,
    check_unexpected_columns,
    check_schema_and_types,
    check_missing_values,
    handle_missing_values
)

# Sample schema for testing
@pytest.fixture
def sample_schema():
    return {
        "timestamp": {"dtype": "datetime64[ns]", "required": True},
        "price": {"dtype": "float64", "required": True}
    }

# Logger fixture to capture logs during testing
@pytest.fixture
def logger():
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    logger.addHandler(handler)
    return logger

def test_validate_data_success(sample_schema, logger):
    # This test provides a fully valid DataFrame
    # Expectation: validation should pass without any errors
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3),
        "price": [1.1, 2.2, 3.3]
    })
    validated_df = validate_data(df.copy(), sample_schema, logger, "drop", "warn")
    assert not validated_df.isnull().any().any()

def test_missing_column_raises(sample_schema, logger):
    # This test omits the 'price' column which is required
    # Expectation: should raise a ValueError because of missing required column
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3)
    })
    with pytest.raises(ValueError):
        validate_data(df, sample_schema, logger, "drop", "raise")

def test_unexpected_column_warns(sample_schema, logger):
    # This test adds an extra column not defined in the schema
    # Expectation: should log a warning, but continue execution
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3),
        "price": [1.0, 2.0, 3.0],
        "extra": ["x", "y", "z"]
    })
    validate_data(df, sample_schema, logger, "drop", "warn")

def test_type_mismatch_warns(sample_schema, logger):
    # This test provides string data in a float column
    # Expectation: should log a warning or type mismatch, not crash
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3),
        "price": ["a", "b", "c"]
    })
    validate_data(df, sample_schema, logger, "drop", "warn")

def test_handle_missing_values_drop(logger):
    # This test ensures rows with missing values are dropped
    # Expectation: only 1 row without missing values remains
    df = pd.DataFrame({
        "a": [1, None, 3],
        "b": [4, 5, None]
    })
    cleaned = handle_missing_values(df, "drop", logger)
    assert cleaned.shape[0] == 1

def test_handle_missing_values_impute(logger):
    # This test ensures missing values are forward/backward filled
    # Expectation: no missing values remain after imputation
    df = pd.DataFrame({
        "a": [1, None, None, 4],
        "b": [None, 2, 3, None]
    })
    cleaned = handle_missing_values(df, "impute", logger)
    assert not cleaned.isnull().any().any()
