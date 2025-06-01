import pandas as pd
import yaml
import json
import os
import logging
from typing import Dict, Optional

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


def validate_data(df: pd.DataFrame, schema: Dict,
                  logger: Optional[logging.Logger] = None,
                  missing_strategy: str = 'drop',
                  on_error: str = 'raise') -> pd.DataFrame:

    # Setup logger if not provided
    if logger is None:
        logger = setup_logging()

    try:
        # Input validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if df.empty:
            logger.warning("Input DataFrame is empty")
            return df

        if not isinstance(schema, dict):
            raise TypeError("Schema must be a dictionary")

        if not schema:
            logger.warning("Schema is empty, no validation performed")
            return df

        shape_msg = f"Starting validation for DataFrame with shape {df.shape}"
        logger.info(shape_msg)
        settings_msg = (f"Validation: missing_strategy='{missing_strategy}', "
                        f"on_error='{on_error}'")
        logger.info(settings_msg)

        # Initialize validation report
        report = {
            'missing_columns': [],
            'unexpected_columns': [],
            'type_mismatches': {},
            'missing_values': {},
            'out_of_range': {}
        }

        # Run validation checks
        check_unexpected_columns(df, schema, logger, on_error, report)
        df_processed = check_schema_and_types(
            df, schema, logger, on_error, report
        )
        check_missing_values(df_processed, schema, logger, report)

        # Handle missing values
        df_final = handle_missing_values(
            df_processed, missing_strategy, logger
        )

        # Save validation report
        save_validation_report(report, logger)

        # Log summary
        total_issues = (len(report['missing_columns']) +
                        len(report['unexpected_columns']) +
                        len(report['type_mismatches']))

        if total_issues == 0:
            success_msg = "✓ Success: Data validation completed with no issues"
            logger.info(success_msg)
        else:
            issues_msg = (f"Data validation completed with {total_issues} "
                          f"issues (see report for details)")
            logger.info(issues_msg)

        logger.info(f"Final DataFrame shape: {df_final.shape}")
        return df_final

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise
