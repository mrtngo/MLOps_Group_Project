import logging
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest
import yaml
from unittest import mock
from src.mlops.data_validation import data_validation

from mlops.data_validation.data_validation import (
    check_missing_values,
    check_schema_and_types,
    check_unexpected_columns,
    check_value_ranges,
    handle_missing_values,
    load_config,
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
    schema_local = {
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
    logger_local = data_validation.setup_logging()
    report = {"missing_columns": [], "unexpected_columns": [], "type_mismatches": {}, "missing_values": {}}
    check_schema_and_types(df, schema_local, logger_local, "warn", report)
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
    # Using default path, ensure no file in tmp
    os.chdir(tmp_path)
    save_validation_report(report, logger)
    # Default creates reports/validation_report.json
    assert (tmp_path / "reports" / "validation_report.json").exists()


def test_validate_data_all_valid(schema):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3),
            "price": [50.0, 75.0, 90.0],
        }
    )
    config = {"schema": {"columns": [{"name": k, **v} for k, v in schema.items()]}, "missing_values_strategy": "drop", "on_error": "warn"}
    validated, report = validate_data(df, config)
    assert not validated.empty
    assert report["status"] == "pass"


def test_validate_data_warn_on_range(schema):
    df = pd.DataFrame(
        {"timestamp": pd.date_range("2024-01-01", periods=2), "price": [5.0, 500000.0]}
    )
    config = {"schema": {"columns": [{"name": k, **v} for k, v in schema.items()]}, "missing_values_strategy": "drop", "on_error": "warn"}
    validated, report = validate_data(df, config)
    assert not validated.empty


def test_check_unexpected_columns_raise():
    df = pd.DataFrame({"a": [1], "b": [2]})
    schema_local = {"a": {"dtype": "int"}}
    logger_local = data_validation.setup_logging()
    report = {}
    with pytest.raises(ValueError):
        check_unexpected_columns(df, schema_local, logger_local, on_error="raise", report=report)


def test_check_value_ranges_out_of_range_raise():
    df = pd.DataFrame({"a": [1, 100]})
    props = {"min": 0, "max": 10}
    logger_local = data_validation.setup_logging()
    report = {}
    with pytest.raises(ValueError):
        check_value_ranges(df, "a", props, logger_local, on_error="raise", report=report)


def test_check_schema_and_types_missing_required():
    df = pd.DataFrame({"a": [1]})
    schema_local = {"a": {"dtype": "int"}, "b": {"dtype": "int", "required": True}}
    logger_local = data_validation.setup_logging()
    report = {"missing_columns": [], "type_mismatches": {}}
    with pytest.raises(ValueError):
        check_schema_and_types(df, schema_local, logger_local, on_error="raise", report=report)


def test_handle_missing_values_unknown_strategy():
    df = pd.DataFrame({"a": [1, None]})
    logger_local = data_validation.setup_logging()
    result = handle_missing_values(df, "unknown", logger_local)
    assert result.equals(df)


def test_save_validation_report_oserror():
    logger_local = data_validation.setup_logging()
    report = {}
    with mock.patch("builtins.open", side_effect=OSError("fail")):
        with pytest.raises(OSError):
            save_validation_report(report, logger_local, output_path="/invalid_dir/report.json")


# ───────────── Additional Tests for Coverage ─────────────

def test_check_value_ranges_partial_violation():
    df = pd.DataFrame({"price": [5.0, 500.0, 10000.0]})
    props = {"min": 10.0, "max": 1000.0, "on_error": "warn"}
    logger_local = data_validation.setup_logging()
    report = {}
    check_value_ranges(df, "price", props, logger_local, on_error="warn", report=report)
    assert "price" in report.get("out_of_range", {})
    assert report["out_of_range"]["price"]["count"] == 2


def test_check_schema_and_types_optional_column():
    schema_local = {
        "a": {"dtype": "int64", "required": True},
        "b": {"dtype": "float64", "required": False},
    }
    df = pd.DataFrame({"a": [1, 2, 3]})
    logger_local = data_validation.setup_logging()
    report = {"missing_columns": [], "type_mismatches": {}}
    df_checked = check_schema_and_types(df, schema_local, logger_local, "warn", report)
    assert "b" not in report["missing_columns"]


def test_validate_data_wrong_dtype():
    config = {
        "schema": {"columns": [
            {"name": "timestamp", "dtype": "float64"},
            {"name": "price", "dtype": "float64"},
        ]},
        "missing_values_strategy": "drop",
        "on_error": "warn"
    }
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=2),
        "price": [100.0, 200.0],
    })
    validated_df, report = validate_data(df, config)
    assert report["status"] == "fail"
    assert any("wrong type" in e for e in report["issues"]["errors"])


def test_check_value_ranges_missing_column():
    df = pd.DataFrame({"a": [1, 2, 3]})
    props = {"min": 0, "max": 10}
    logger_local = data_validation.setup_logging()
    report = {}
    with pytest.raises(KeyError):
        check_value_ranges(df, "missing_col", props, logger_local, on_error="raise", report=report)


def test_load_config_invalid_yaml(tmp_path):
    bad_yaml = tmp_path / "bad_config.yaml"
    bad_yaml.write_text("not: valid: yaml: [")
    logger_local = data_validation.setup_logging()
    with pytest.raises(yaml.YAMLError):
        load_config(str(bad_yaml), logger_local)


def test_load_config_not_dict(tmp_path):
    file = tmp_path / "bad.yaml"
    file.write_text("[]")
    logger_local = data_validation.setup_logging()
    with pytest.raises(ValueError):
        load_config(str(file), logger_local)
