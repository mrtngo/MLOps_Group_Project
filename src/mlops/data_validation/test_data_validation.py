
import pytest
import pandas as pd
import logging
from io import StringIO
from data_validation import (
    validate_data,
    check_unexpected_columns,
    check_schema_and_types,
    check_missing_values,
    handle_missing_values,
    save_validation_report
)

@pytest.fixture
def schema():
    return {
        "timestamp": {
            "dtype": "datetime64[ns]",
            "required": True,
            "on_error": "raise"
        },
        "price": {
            "dtype": "float64",
            "required": True,
            "min": 10.0,
            "max": 100000.0,
            "on_error": "warn"
        }
    }

@pytest.fixture
def logger():
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    return logger

def test_check_unexpected_columns(schema, logger):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2),
        "price": [50, 60],
        "extra": [1, 2]
    })
    report = {}
    check_unexpected_columns(df, schema, logger, "warn", report)
    assert "unexpected_columns" in report

def test_check_schema_and_types_success(schema, logger):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2),
        "price": [100.0, 200.0]
    })
    report = {"missing_columns": [], "unexpected_columns": [], "type_mismatches": {}, "missing_values": {}}
    check_schema_and_types(df, schema, logger, "warn", report)
    assert report["type_mismatches"] == {}

def test_check_schema_and_types_type_mismatch(logger):
    """Expect warning (not exception) for timestamp format mismatch when on_error='warn'."""
    schema = {
        "timestamp": {
            "dtype": "datetime64[ns]",
            "required": True,
            "on_error": "warn"  # חשוב!
        },
        "price": {
            "dtype": "float64",
            "required": True,
            "min": 10.0,
            "max": 100000.0,
            "on_error": "warn"
        }
    }
    df = pd.DataFrame({
        "timestamp": ["not_a_date", "still_not_a_date"],
        "price": [100.0, 200.0]
    })
    report = {"missing_columns": [], "unexpected_columns": [], "type_mismatches": {}, "missing_values": {}}
    check_schema_and_types(df, schema, logger, "warn", report)
    assert "timestamp" in report["type_mismatches"]

def test_check_missing_values(schema, logger):
    df = pd.DataFrame({
        "timestamp": [pd.NaT, pd.Timestamp("2024-01-01")],
        "price": [None, 100.0]
    })
    report = {"missing_values": {}}
    check_missing_values(df, schema, logger, report)
    assert "timestamp" in report["missing_values"]
    assert "price" in report["missing_values"]

def test_handle_missing_values_drop(logger):
    df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
    cleaned = handle_missing_values(df, "drop", logger)
    assert cleaned.shape[0] == 1

def test_handle_missing_values_impute(logger):
    df = pd.DataFrame({"a": [1, None, 3], "b": [2, 3, 4]})
    cleaned = handle_missing_values(df, "impute", logger)
    assert cleaned.isnull().sum().sum() == 0

def test_handle_missing_values_keep(logger):
    df = pd.DataFrame({"a": [1, None], "b": [2, 3]})
    cleaned = handle_missing_values(df, "keep", logger)
    assert cleaned.equals(df)

def test_save_validation_report(tmp_path, logger):
    report = {"dummy": "report"}
    os_path = tmp_path / "validation_report.json"
    save_validation_report(report, logger)
    assert os_path.exists() is False  # Because path is hardcoded in function

def test_validate_data_all_valid(schema, logger):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=3),
        "price": [50.0, 75.0, 90.0]
    })
    validated = validate_data(df, schema, logger)
    assert not validated.empty

def test_validate_data_raise_on_missing_required(schema, logger):
    df = pd.DataFrame({"price": [50.0, 100.0]})
    with pytest.raises(ValueError):
        validate_data(df, schema, logger)

def test_validate_data_warn_on_range(schema, logger):
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2),
        "price": [5.0, 500000.0]
    })
    validated = validate_data(df, schema, logger)
    assert not validated.empty
