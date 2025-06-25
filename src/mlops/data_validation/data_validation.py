import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ───────────────────────────── setup logging ──────────────────────────────


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration for data validation."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_config(config_path: str, logger: Optional[logging.Logger] = None) -> Dict:
    if logger is None:
        logger = setup_logging()

    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info("Loading configuration from: %s", config_path)

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a valid dictionary")

        logger.info("Configuration loaded successfully")
        return config

    except FileNotFoundError:
        logger.error("Configuration file not found: %s", config_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing YAML configuration: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error loading configuration: %s", e)
        raise


def check_unexpected_columns(
    df: pd.DataFrame, schema: Dict, logger: logging.Logger, on_error: str, report: Dict
) -> None:
    try:
        logger.debug("Checking for unexpected columns")

        if not isinstance(schema, dict):
            logger.error("Schema must be a dictionary")
            raise ValueError("Schema must be a dictionary")

        expected_columns = set(schema.keys())
        actual_columns = set(df.columns)
        unexpected_cols = actual_columns - expected_columns

        if unexpected_cols:
            report["unexpected_columns"] = list(unexpected_cols)
            msg = "Found %d unexpected columns: %s" % (
                len(unexpected_cols),
                sorted(unexpected_cols),
            )
            logger.info("%s", msg)

            if on_error == "raise":
                error_msg = "Unexpected columns found: %s" % unexpected_cols
                logger.error("%s", error_msg)
                raise ValueError(error_msg)
            else:
                logger.warning("Continuing despite unexpected columns")
        else:
            logger.debug("No unexpected columns found")

    except Exception as e:
        logger.error(f"Error checking unexpected columns: {e}")
        raise


def check_value_ranges(
    df: pd.DataFrame,
    col: str,
    props: Dict,
    logger: logging.Logger,
    on_error: str,
    report: Dict,
) -> None:
    try:
        if "min" in props or "max" in props:
            logger.debug(f"Checking value ranges for column '{col}'")

            min_val = props.get("min")
            max_val = props.get("max")

            # Build conditions safely
            conditions = []
            if min_val is not None:
                conditions.append(df[col] < min_val)
            if max_val is not None:
                conditions.append(df[col] > max_val)

            if conditions:
                # Combine conditions with OR
                combined_condition = conditions[0]
                for condition in conditions[1:]:
                    combined_condition = combined_condition | condition

                out_of_range = df[col][combined_condition]

                if not out_of_range.empty:
                    report.setdefault("out_of_range", {})[col] = {
                        "count": len(out_of_range),
                        "min_allowed": min_val,
                        "max_allowed": max_val,
                        "actual_min": (
                            float(df[col].min()) if not df[col].empty else None
                        ),
                        "actual_max": (
                            float(df[col].max()) if not df[col].empty else None
                        ),
                    }

                    msg = "Column '%s' has %d values out of range [%s, %s]" % (
                        col,
                        len(out_of_range),
                        min_val,
                        max_val,
                    )
                    logger.info("%s", msg)

                    col_error = props.get("on_error", on_error)
                    if col_error == "raise":
                        error_msg = (
                            "Raising error for out-of-range values in column '%s'" % col
                        )
                        logger.error("%s", error_msg)
                        raise ValueError(msg)
                    else:
                        warn_msg = (
                            "Continuing despite out-of-range values in column '%s'"
                            % col
                        )
                        logger.warning("%s", warn_msg)
                else:
                    range_msg = "All values in col '%s' are within range" % col
                    logger.debug("%s", range_msg)

    except KeyError as e:
        logger.error(f"Column '{col}' not found in DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"Error checking value ranges for column '{col}': {e}")
        raise


def check_schema_and_types(
    df: pd.DataFrame, schema: Dict, logger: logging.Logger, on_error: str, report: Dict
) -> pd.DataFrame:
    try:
        logger.info("Validating schema for %d columns", len(schema))
        df_copy = df.copy()  # Work on a copy to avoid modifying original

        for col, props in schema.items():
            try:
                logger.debug(f"Processing column '{col}'")
                col_error = props.get("on_error", on_error)

                # Check if column exists
                if col not in df_copy.columns:
                    if props.get("required", True):
                        report["missing_columns"].append(col)
                        msg = "Missing required column: '%s'" % col
                        logger.warning("%s", msg)

                        if col_error == "raise":
                            error_msg = (
                                "Raising error for missing required column '%s'" % col
                            )
                            logger.error("%s", error_msg)
                            raise ValueError(msg)
                    else:
                        optional_msg = "Optional col '%s' not found, skip" % col
                        logger.info("%s", optional_msg)
                    continue

                # Type validation and conversion
                expected_type = props.get("dtype")
                if expected_type:
                    actual_type = str(df_copy[col].dtype)
                    debug_msg = "Column '%s': expected %s, actual %s" % (
                        col,
                        expected_type,
                        actual_type,
                    )
                    logger.debug("%s", debug_msg)

                    if expected_type == "datetime64[ns]":
                        try:
                            df_copy[col] = pd.to_datetime(df_copy[col])
                            success_msg = (
                                "Successfully converted column '%s' to datetime" % col
                            )
                            logger.debug("%s", success_msg)
                        except Exception as e:
                            report["type_mismatches"][col] = {
                                "expected": expected_type,
                                "actual": actual_type,
                                "error": str(e),
                            }
                            msg = "Failed to convert column '%s' to datetime: %s" % (
                                col,
                                e,
                            )
                            logger.warning("%s", msg)

                            if col_error == "raise":
                                error_msg = (
                                    "Raising error for datetime conversion failure in column '%s'"
                                    % col
                                )
                                logger.error("%s", error_msg)
                                raise

                    elif expected_type.startswith(
                        "float"
                    ) and not pd.api.types.is_float_dtype(df_copy[col]):
                        try:
                            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
                            success_msg = (
                                "Successfully converted column '%s' to float" % col
                            )
                            logger.debug("%s", success_msg)
                        except Exception as e:
                            report["type_mismatches"][col] = {
                                "expected": expected_type,
                                "actual": actual_type,
                                "error": str(e),
                            }
                            msg = "Failed to convert column '%s' to float: %s" % (
                                col,
                                e,
                            )
                            logger.warning("%s", msg)

                            if col_error == "raise":
                                error_msg = (
                                    "Raising error for float conversion failure in column '%s'"
                                    % col
                                )
                                logger.error("%s", error_msg)
                                raise

                # Range validation
                check_value_ranges(df_copy, col, props, logger, on_error, report)

            except Exception as e:
                logger.error("Error processing column '%s': %s", col, e)
                if col_error == "raise":
                    raise
                continue

        logger.info("Schema validation completed")
        return df_copy

    except Exception as e:
        logger.error("Error in schema validation: %s", e)
        raise


def check_missing_values(
    df: pd.DataFrame, schema: Dict, logger: logging.Logger, report: Dict
) -> None:
    try:
        logger.debug("Checking for missing values")
        total_missing = 0

        for col in schema.keys():
            if col in df.columns:
                try:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        report["missing_values"][col] = int(missing_count)
                        missing_pct = (missing_count / len(df)) * 100
                        msg = "Column '%s': %d missing values (%.1f%%)" % (
                            col,
                            missing_count,
                            missing_pct,
                        )
                        logger.info("%s", msg)
                        total_missing += missing_count
                except Exception as e:
                    error_msg = "Error checking missing values for column '%s': %s" % (
                        col,
                        e,
                    )
                    logger.error("%s", error_msg)
                    continue

        if total_missing == 0:
            logger.info("No missing values found")
        else:
            total_msg = "Total missing values in all columns: %d" % total_missing
            logger.info("%s", total_msg)

    except Exception as e:
        logger.error("Error checking missing values: %s", e)
        raise


def handle_missing_values(
    df: pd.DataFrame, strategy: str, logger: logging.Logger
) -> pd.DataFrame:
    try:
        original_shape = df.shape
        logger.info("Handling missing values with strategy: '%s'", strategy)

        if strategy == "drop":
            result_df = df.dropna()
            msg = "Dropped rows with missing values: %d -> %d rows" % (
                original_shape[0],
                result_df.shape[0],
            )
            logger.info("%s", msg)

        elif strategy == "impute":
            result_df = df.copy()
            # Forward fill, then backward fill
            result_df = result_df.ffill().bfill()
            imputed_count = df.isnull().sum().sum() - result_df.isnull().sum().sum()
            msg = (
                "Imputed %d missing values using forward/backward fill" % imputed_count
            )
            logger.info("%s", msg)

        elif strategy == "keep":
            result_df = df.copy()
            logger.info("Keeping all missing values as-is")

        else:
            warn_msg = (
                "Unknown missing_values_strategy: '%s'. Keeping data unchanged."
                % strategy
            )
            logger.warning("%s", warn_msg)
            result_df = df.copy()

        return result_df

    except Exception as e:
        error_msg = "Error handling missing values with strategy '%s': %s" % (
            strategy,
            e,
        )
        logger.error("%s", error_msg)
        raise


def save_validation_report(
    report: Dict,
    logger: logging.Logger,
    output_path: str = "reports/validation_report.json",
) -> None:

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Add summary statistics to report
        total_issues = (
            len(report.get("missing_columns", []))
            + len(report.get("unexpected_columns", []))
            + len(report.get("type_mismatches", {}))
            + sum(report.get("missing_values", {}).values())
            + sum(
                len(v) if isinstance(v, list) else 1
                for v in report.get("out_of_range", {}).values()
            )
        )

        validation_passed = (
            len(report.get("missing_columns", [])) == 0
            and len(report.get("type_mismatches", {})) == 0
        )

        report["summary"] = {
            "total_issues": total_issues,
            "validation_passed": validation_passed,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("Validation report saved to: %s", output_path)

    except OSError as e:
        logger.error("Failed to create directory or write file: %s", e)
        raise
    except Exception as e:
        logger.error("Error saving validation report: %s", e)
        raise


def validate_data(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Validates the input DataFrame based on a schema defined in the config.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        config (Dict): The data validation configuration.

    Returns:
        Tuple[pd.DataFrame, Dict]: A tuple containing the validated (and possibly imputed)
                                   DataFrame and a validation report dictionary.
    """
    report = {
        "status": "pass",
        "issues": {"errors": [], "warnings": []},
        "column_details": {},
        "missing_values_summary": None,
    }

    schema_config = config.get("schema", {})
    expected_cols = {col["name"]: col for col in schema_config.get("columns", [])}
    df_cols = set(df.columns)

    logger.info("Starting validation for DataFrame with shape %s", df.shape)
    logger.info(
        "Validation: missing_strategy='%s', on_error='%s'",
        config.get("missing_values_strategy"),
        config.get("on_error"),
    )

    # Check for missing and unexpected columns
    for col_name in expected_cols:
        if col_name not in df_cols:
            msg = "Missing expected column: '%s'" % col_name
            report["issues"]["errors"].append(msg)
            report["status"] = "fail"
            logger.error("%s", msg)

    unexpected_cols = df_cols - set(expected_cols.keys())
    if unexpected_cols:
        msg = "Found %d unexpected columns: %s" % (
            len(unexpected_cols),
            list(unexpected_cols),
        )
        report["issues"]["warnings"].append(msg)
        logger.warning("%s", msg)

    if report["status"] == "fail":
        return df, report  # Stop further validation if essential columns are missing

    # Validate schema for each column
    logger.info("Validating schema for %d columns", len(expected_cols))
    for name, params in expected_cols.items():
        column_report = {"expected_type": params["dtype"], "status": "pass"}

        # Check type
        if df[name].dtype.name != params["dtype"]:
            msg = "Column '%s' has wrong type. Expected %s, found %s" % (
                name,
                params["dtype"],
                df[name].dtype.name,
            )
            report["issues"]["errors"].append(msg)
            report["status"] = "fail"
            column_report["status"] = "fail"
            logger.error("%s", msg)

        # Get sample values
        sample_values = df[name].dropna().unique()[:5]
        column_report["sample_values"] = [
            str(v) for v in sample_values
        ]  # Convert to string for JSON

        report["column_details"][name] = column_report

    logger.info("Schema validation completed")

    # Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        missing_strategy = config.get("missing_values_strategy", "drop")
        summary = {"strategy": missing_strategy, "missing_before": int(missing_before)}

        if missing_strategy == "impute":
            df = df.fillna(method="ffill").fillna(method="bfill")
            missing_after = df.isnull().sum().sum()
            imputed_count = missing_before - missing_after
            summary["total_imputed"] = int(imputed_count)
            logger.info(
                "Imputed %d missing values using forward/backward fill", imputed_count
            )
        elif missing_strategy == "drop":
            rows_before = len(df)
            df.dropna(inplace=True)
            rows_after = len(df)
            dropped_count = rows_before - rows_after
            summary["rows_dropped"] = int(dropped_count)
            logger.info("Dropped %d rows with missing values", dropped_count)

        report["missing_values_summary"] = summary

    logger.info(
        "Data validation completed with %d errors and %d warnings.",
        len(report["issues"]["errors"]),
        len(report["issues"]["warnings"]),
    )

    return df, report
