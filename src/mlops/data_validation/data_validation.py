import pandas as pd
import yaml
import json
import os
import logging
from typing import Dict, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ───────────────────────────── setup logging ──────────────────────────────


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration for data validation."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str,
                logger: Optional[logging.Logger] = None) -> Dict:
    if logger is None:
        logger = setup_logging()

    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        logger.info(f"Loading configuration from: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(
                "Configuration file must contain a valid dictionary"
            )

        logger.info("Configuration loaded successfully")
        return config

    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}")
        raise


def check_unexpected_columns(df: pd.DataFrame, schema: Dict,
                             logger: logging.Logger, on_error: str,
                             report: Dict) -> None:
    try:
        logger.debug("Checking for unexpected columns")

        if not isinstance(schema, dict):
            logger.error("Schema must be a dictionary")
            raise ValueError("Schema must be a dictionary")

        expected_columns = set(schema.keys())
        actual_columns = set(df.columns)
        unexpected_cols = actual_columns - expected_columns

        if unexpected_cols:
            report['unexpected_columns'] = list(unexpected_cols)
            msg = (f"Found {len(unexpected_cols)} unexpected columns: "
                   f"{sorted(unexpected_cols)}")
            logger.info(msg)

            if on_error == 'raise':
                logger.error("Raising error due to unexpected columns")
                error_msg = f"Unexpected columns found: {unexpected_cols}"
                raise ValueError(error_msg)
            else:
                logger.warning("Continuing despite unexpected columns")
        else:
            logger.debug("No unexpected columns found")

    except Exception as e:
        logger.error(f"Error checking unexpected columns: {e}")
        raise


def check_value_ranges(df: pd.DataFrame, col: str, props: Dict,
                       logger: logging.Logger, on_error: str,
                       report: Dict) -> None:
    try:
        if 'min' in props or 'max' in props:
            logger.debug(f"Checking value ranges for column '{col}'")

            min_val = props.get('min')
            max_val = props.get('max')

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
                    report.setdefault('out_of_range', {})[col] = {
                        'count': len(out_of_range),
                        'min_allowed': min_val,
                        'max_allowed': max_val,
                        'actual_min': (float(df[col].min())
                                       if not df[col].empty else None),
                        'actual_max': (float(df[col].max())
                                       if not df[col].empty else None)
                    }

                    msg = (f"Column '{col}' has {len(out_of_range)} values "
                           f"out of range [{min_val}, {max_val}]")
                    logger.info(msg)

                    col_error = props.get('on_error', on_error)
                    if col_error == 'raise':
                        error_msg = (f"Raising error for out-of-range values "
                                     f"in column '{col}'")
                        logger.error(error_msg)
                        raise ValueError(msg)
                    else:
                        warn_msg = (f"Continuing despite out-of-range values "
                                    f"in column '{col}'")
                        logger.warning(warn_msg)
                else:
                    range_msg = f"All values in col '{col}' are within range"
                    logger.debug(range_msg)

    except KeyError as e:
        logger.error(f"Column '{col}' not found in DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"Error checking value ranges for column '{col}': {e}")
        raise


def check_schema_and_types(df: pd.DataFrame, schema: Dict,
                           logger: logging.Logger, on_error: str,
                           report: Dict) -> pd.DataFrame:
    try:
        logger.info(f"Validating schema for {len(schema)} columns")
        df_copy = df.copy()  # Work on a copy to avoid modifying original

        for col, props in schema.items():
            try:
                logger.debug(f"Processing column '{col}'")
                col_error = props.get('on_error', on_error)

                # Check if column exists
                if col not in df_copy.columns:
                    if props.get('required', True):
                        report['missing_columns'].append(col)
                        msg = f"Missing required column: '{col}'"
                        logger.warning(msg)

                        if col_error == 'raise':
                            error_msg = (f"Raising error for missing required "
                                         f"column '{col}'")
                            logger.error(error_msg)
                            raise ValueError(msg)
                    else:
                        optional_msg = f"Optional col '{col}' not found, skip"
                        logger.info(optional_msg)
                    continue

                # Type validation and conversion
                expected_type = props.get('dtype')
                if expected_type:
                    actual_type = str(df_copy[col].dtype)
                    debug_msg = (f"Column '{col}': expected {expected_type}, "
                                 f"actual {actual_type}")
                    logger.debug(debug_msg)

                    if expected_type == 'datetime64[ns]':
                        try:
                            df_copy[col] = pd.to_datetime(df_copy[col])
                            success_msg = (f"Successfully converted column "
                                           f"'{col}' to datetime")
                            logger.debug(success_msg)
                        except Exception as e:
                            report['type_mismatches'][col] = {
                                'expected': expected_type,
                                'actual': actual_type,
                                'error': str(e)
                            }
                            msg = (f"Failed to convert column '{col}' "
                                   f"to datetime: {e}")
                            logger.warning(msg)

                            if col_error == 'raise':
                                error_msg = (f"Raising error for datetime "
                                             f"conversion failure in column "
                                             f"'{col}'")
                                logger.error(error_msg)
                                raise

                    elif (expected_type.startswith('float') and
                          not pd.api.types.is_float_dtype(df_copy[col])):
                        try:
                            df_copy[col] = pd.to_numeric(
                                df_copy[col], errors='coerce'
                            )
                            success_msg = (f"Successfully converted column "
                                           f"'{col}' to float")
                            logger.debug(success_msg)
                        except Exception as e:
                            report['type_mismatches'][col] = {
                                'expected': expected_type,
                                'actual': actual_type,
                                'error': str(e)
                            }
                            msg = (f"Failed to convert column '{col}' "
                                   f"to float: {e}")
                            logger.warning(msg)

                            if col_error == 'raise':
                                error_msg = (f"Raising error for float "
                                             f"conversion failure in column "
                                             f"'{col}'")
                                logger.error(error_msg)
                                raise

                # Range validation
                check_value_ranges(
                    df_copy, col, props, logger, on_error, report
                )

            except Exception as e:
                logger.error(f"Error processing column '{col}': {e}")
                if col_error == 'raise':
                    raise
                continue

        logger.info("Schema validation completed")
        return df_copy

    except Exception as e:
        logger.error(f"Error in schema validation: {e}")
        raise


def check_missing_values(df: pd.DataFrame, schema: Dict,
                         logger: logging.Logger, report: Dict) -> None:
    try:
        logger.debug("Checking for missing values")
        total_missing = 0

        for col in schema.keys():
            if col in df.columns:
                try:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        report['missing_values'][col] = int(missing_count)
                        missing_pct = (missing_count / len(df)) * 100
                        msg = (f"Column '{col}': {missing_count} missing "
                               f"values ({missing_pct:.1f}%)")
                        logger.info(msg)
                        total_missing += missing_count
                except Exception as e:
                    error_msg = (f"Error checking missing values for "
                                 f"column '{col}': {e}")
                    logger.error(error_msg)
                    continue

        if total_missing == 0:
            logger.info("No missing values found")
        else:
            total_msg = f"Total missing values in all columns: {total_missing}"
            logger.info(total_msg)

    except Exception as e:
        logger.error(f"Error checking missing values: {e}")
        raise


def handle_missing_values(df: pd.DataFrame, strategy: str,
                          logger: logging.Logger) -> pd.DataFrame:
    try:
        original_shape = df.shape
        logger.info(f"Handling missing values with strategy: '{strategy}'")

        if strategy == 'drop':
            result_df = df.dropna()
            msg = (f"Dropped rows with missing values: {original_shape[0]} -> "
                   f"{result_df.shape[0]} rows")
            logger.info(msg)

        elif strategy == 'impute':
            result_df = df.copy()
            # Forward fill, then backward fill
            result_df = result_df.ffill().bfill()
            imputed_count = (df.isnull().sum().sum() -
                             result_df.isnull().sum().sum())
            msg = (f"Imputed {imputed_count} missing values using "
                   f"forward/backward fill")
            logger.info(msg)

        elif strategy == 'keep':
            result_df = df.copy()
            logger.info("Keeping all missing values as-is")

        else:
            warn_msg = (f"Unknown missing_values_strategy: '{strategy}'. "
                        f"Keeping data unchanged.")
            logger.warning(warn_msg)
            result_df = df.copy()

        return result_df

    except Exception as e:
        error_msg = (f"Error handling missing values with strategy "
                     f"'{strategy}': {e}")
        logger.error(error_msg)
        raise


def save_validation_report(
        report: Dict, logger: logging.Logger,
        output_path: str = 'reports/validation_report.json') -> None:

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Add summary statistics to report
        total_issues = (
            len(report.get('missing_columns', [])) +
            len(report.get('unexpected_columns', [])) +
            len(report.get('type_mismatches', {})) +
            sum(report.get('missing_values', {}).values()) +
            sum(len(v) if isinstance(v, list) else 1
                for v in report.get('out_of_range', {}).values())
        )

        validation_passed = (len(report.get('missing_columns', [])) == 0 and
                             len(report.get('type_mismatches', {})) == 0)

        report['summary'] = {
            'total_issues': total_issues,
            'validation_passed': validation_passed
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report saved to: {output_path}")

    except OSError as e:
        logger.error(f"Failed to create directory or write file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error saving validation report: {e}")
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
        "missing_values_summary": None
    }
    
    schema_config = config.get("schema", {})
    expected_cols = {col['name']: col for col in schema_config.get('columns', [])}
    df_cols = set(df.columns)

    logger.info(f"Starting validation for DataFrame with shape {df.shape}")
    logger.info(f"Validation: missing_strategy='{config.get('missing_values_strategy')}', on_error='{config.get('on_error')}'")

    # Check for missing and unexpected columns
    for col_name in expected_cols:
        if col_name not in df_cols:
            msg = f"Missing expected column: '{col_name}'"
            report["issues"]["errors"].append(msg)
            report["status"] = "fail"
            logger.error(msg)
    
    unexpected_cols = df_cols - set(expected_cols.keys())
    if unexpected_cols:
        msg = f"Found {len(unexpected_cols)} unexpected columns: {list(unexpected_cols)}"
        report["issues"]["warnings"].append(msg)
        logger.warning(f"Continuing despite unexpected columns")
    
    if report["status"] == "fail":
        return df, report # Stop further validation if essential columns are missing

    # Validate schema for each column
    logger.info(f"Validating schema for {len(expected_cols)} columns")
    for name, params in expected_cols.items():
        column_report = {"expected_type": params["dtype"], "status": "pass"}
        
        # Check type
        if df[name].dtype.name != params["dtype"]:
            msg = f"Column '{name}' has wrong type. Expected {params['dtype']}, found {df[name].dtype.name}"
            report["issues"]["errors"].append(msg)
            report["status"] = "fail"
            column_report["status"] = "fail"
            logger.error(msg)
        
        # Get sample values
        sample_values = df[name].dropna().unique()[:5]
        column_report["sample_values"] = [str(v) for v in sample_values] # Convert to string for JSON
        
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
            logger.info(f"Imputed {imputed_count} missing values using forward/backward fill")
        elif missing_strategy == "drop":
            rows_before = len(df)
            df.dropna(inplace=True)
            rows_after = len(df)
            dropped_count = rows_before - rows_after
            summary["rows_dropped"] = int(dropped_count)
            logger.info(f"Dropped {dropped_count} rows with missing values")
        
        report["missing_values_summary"] = summary

    logger.info(f"Data validation completed with {len(report['issues']['errors'])} errors and {len(report['issues']['warnings'])} warnings.")
    
    return df, report
