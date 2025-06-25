import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest
from unittest import mock
from src.mlops.data_validation import data_validation

from mlops.data_validation.data_validation import (
    check_missing_values,
    check_schema_and_types,
    check_unexpected_columns,
    handle_missing_values,
    save_validation_report,
    validate_data,
)


@pytest.fixture
def schema():
    return {
        "timestamp": {"dtype": "datetime64[ns]", "required": True, "on_error": "raise"},
        "price": {
            "dtype": "float64",
            "required": True,
            "min": 10.0,
            "max": 100000.0,
            "on_error": "warn",
        },
    }


@pytest.fixture
def logger():
    test_logger = logging.getLogger("test_logger")
    test_logger.setLevel(logging.DEBUG)
    test_logger.addHandler(logging.StreamHandler())
    return test_logger


def test_check_unexpected_columns(schema, logger):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2),
            "price": [50, 60],
            "extra": [1, 2],
        }
    )
    report = {}
    check_unexpected_columns(df, schema, logger, "warn", report)
    assert "unexpected_columns" in report


def test_check_schema_and_types_success(schema, logger):
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=2), "price": [100.0, 200.0]}
    )
    report = {
        "missing_columns": [],
        "unexpected_columns": [],
        "type_mismatches": {},
        "missing_values": {},
    }
    check_schema_and_types(df, schema, logger, "warn", report)
    assert report["type_mismatches"] == {}


def test_check_schema_and_types_type_mismatch():
    """Expect warning (not exception) for timestamp format mismatch
    when on_error='warn'."""
    schema = {
        "timestamp": {"dtype": "datetime64[ns]", "required": True, "on_error": "warn"},
        "price": {
            "dtype": "float64",
            "required": True,
            "min": 10.0,
            "max": 100000.0,
            "on_error": "warn",
        },
    }
    df = pd.DataFrame(
        {"timestamp": ["not_a_date", "still_not_a_date"], "price": [100.0, 200.0]}
    )
    logger = data_validation.setup_logging()
    report = {
        "missing_columns": [],
        "unexpected_columns": [],
        "type_mismatches": {},
        "missing_values": {},
    }
    check_schema_and_types(df, schema, logger, "warn", report)
    assert "timestamp" in report["type_mismatches"]


def test_check_missing_values(schema, logger):
    df = pd.DataFrame(
        {"timestamp": [pd.NaT, pd.Timestamp("2024-01-01")], "price": [None, 100.0]}
    )
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
    # Path is hardcoded in function, so file won't exist at tmp_path
    assert os_path.exists() is False


def test_validate_data_all_valid(schema):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3),
            "price": [50.0, 75.0, 90.0],
        }
    )
    validated, _ = validate_data(df, schema)
    assert not validated.empty


def test_validate_data_warn_on_range(schema):
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=2), "price": [5.0, 500000.0]}
    )
    validated, _ = validate_data(df, schema)
    assert not validated.empty


def test_check_unexpected_columns_raise():
    df = pd.DataFrame({"a": [1], "b": [2]})
    schema = {"a": {"dtype": "int"}}
    logger = data_validation.setup_logging()
    report = {}
    with pytest.raises(ValueError):
        data_validation.check_unexpected_columns(df, schema, logger, on_error="raise", report=report)


def test_check_value_ranges_out_of_range_raise():
    df = pd.DataFrame({"a": [1, 100]})
    props = {"min": 0, "max": 10}
    logger = data_validation.setup_logging()
    report = {}
    with pytest.raises(ValueError):
        data_validation.check_value_ranges(df, "a", props, logger, on_error="raise", report=report)


def test_check_schema_and_types_missing_required():
    df = pd.DataFrame({"a": [1]})
    schema = {"a": {"dtype": "int"}, "b": {"dtype": "int", "required": True}}
    logger = data_validation.setup_logging()
    report = {"missing_columns": [], "type_mismatches": {}}
    with pytest.raises(ValueError):
        data_validation.check_schema_and_types(df, schema, logger, on_error="raise", report=report)


def test_handle_missing_values_unknown_strategy():
    df = pd.DataFrame({"a": [1, None]})
    logger = data_validation.setup_logging()
    result = data_validation.handle_missing_values(df, "unknown", logger)
    assert result.equals(df)


def test_save_validation_report_oserror():
    logger = data_validation.setup_logging()
    report = {}
    with mock.patch("builtins.open", side_effect=OSError("fail")):
        with pytest.raises(OSError):
            data_validation.save_validation_report(report, logger, output_path="/invalid_dir/report.json")
