import pandas as pd
import yaml
import json
import os
from typing import Dict


def load_config(config_path: str) -> Dict:
    """
    Load configuration schema from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary containing schema and settings.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_unexpected_columns(df: pd.DataFrame, schema: Dict, logger, on_error: str, report: Dict):
    """
    Check for unexpected columns that are not defined in the schema.

    Args:
        df (pd.DataFrame): The input DataFrame to validate.
        schema (Dict): Schema definition mapping columns to properties.
        logger: Logger object for logging warnings/errors.
        on_error (str): Global behavior on error ('raise' or 'warn').
        report (Dict): Dictionary to store validation results.
    """
    expected_columns = set(schema.keys())
    actual_columns = set(df.columns)
    unexpected_cols = actual_columns - expected_columns
    if unexpected_cols:
        report['unexpected_columns'] = list(unexpected_cols)
        msg = f"Unexpected columns found: {unexpected_cols}"
        if on_error == 'raise':
            logger.error(msg)
            raise ValueError(msg)
        else:
            logger.warning(msg)


def check_value_ranges(df: pd.DataFrame, col: str, props: Dict, logger, on_error: str, report: Dict):
    """
    Validate whether the values in a given column fall within allowed min/max bounds.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col (str): The name of the column to check.
        props (Dict): Properties of the column, including min/max.
        logger: Logger object.
        on_error (str): Global behavior on error ('raise' or 'warn').
        report (Dict): Dictionary to store validation results.
    """
    if 'min' in props or 'max' in props:
        out_of_range = df[col][
            ((props.get('min') is not None) & (df[col] < props['min'])) |
            ((props.get('max') is not None) & (df[col] > props['max']))
        ]
        if not out_of_range.empty:
            report.setdefault('out_of_range', {})[col] = {
                'count': len(out_of_range),
                'min_allowed': props.get('min'),
                'max_allowed': props.get('max')
            }
            msg = f"Column '{col}' has {len(out_of_range)} values out of allowed range"
            col_error = props.get('on_error', on_error)
            if col_error == 'raise':
                logger.error(msg)
                raise ValueError(msg)
            else:
                logger.warning(msg)


def check_schema_and_types(df: pd.DataFrame, schema: Dict, logger, on_error: str, report: Dict):
    """
    Validate each column in the schema:
    - Required columns exist
    - Data type matches expected
    - Values are within allowed ranges

    Args:
        df (pd.DataFrame): The input DataFrame.
        schema (Dict): Dictionary defining expected schema.
        logger: Logger object.
        on_error (str): Global error handling ('raise' or 'warn').
        report (Dict): Dictionary to update with validation results.
    """
    for col, props in schema.items():
        col_error = props.get('on_error', on_error)

        if col not in df.columns:
            if props.get('required', True):
                report['missing_columns'].append(col)
                msg = f"Missing required column: {col}"
                if col_error == 'raise':
                    logger.error(msg)
                    raise ValueError(msg)
                else:
                    logger.warning(msg)
            continue

        expected_type = props['dtype']
        actual_type = str(df[col].dtype)
        if expected_type == 'datetime64[ns]':
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                report['type_mismatches'][col] = {'expected': expected_type, 'actual': actual_type}
                msg = f"Failed to convert column '{col}' to datetime: {e}"
                if col_error == 'raise':
                    logger.error(msg)
                    raise
                else:
                    logger.warning(msg)
        elif expected_type.startswith('float') and not pd.api.types.is_float_dtype(df[col]):
            report['type_mismatches'][col] = {'expected': expected_type, 'actual': actual_type}
            msg = f"Type mismatch in column '{col}': expected {expected_type}, got {actual_type}"
            if col_error == 'raise':
                logger.error(msg)
                raise TypeError(msg)
            else:
                logger.warning(msg)

        check_value_ranges(df, col, props, logger, on_error, report)


def check_missing_values(df: pd.DataFrame, schema: Dict, logger, report: Dict):
    """
    Check and report the number of missing values for each column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        schema (Dict): Dictionary defining schema columns.
        logger: Logger object.
        report (Dict): Dictionary to update with missing value info.
    """
    for col in schema.keys():
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                report['missing_values'][col] = int(missing_count)
                logger.warning(f"Column '{col}' has {missing_count} missing values.")


def handle_missing_values(df: pd.DataFrame, strategy: str, logger) -> pd.DataFrame:
    """
    Handle missing values using a specific strategy.

    Args:
        df (pd.DataFrame): The input DataFrame.
        strategy (str): Strategy to use ('drop', 'impute', 'keep').
        logger: Logger object.

    Returns:
        pd.DataFrame: Cleaned or unchanged DataFrame depending on strategy.
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'impute':
        return df.ffill().bfill()
    elif strategy == 'keep':
        return df
    else:
        logger.warning(f"Unknown missing_values_strategy: {strategy}. Proceeding without changes.")
        return df


def save_validation_report(report: Dict, logger):
    """
    Save the validation report as a JSON file.

    Args:
        report (Dict): The report dictionary to save.
        logger: Logger object.
    """
    os.makedirs('reports', exist_ok=True)
    with open('reports/validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved to 'reports/validation_report.json'")


def validate_data(df: pd.DataFrame, schema: Dict, logger, missing_strategy: str = 'drop', on_error: str = 'raise') -> pd.DataFrame:
    """
    Main entry point for data validation.

    This function performs:
    - Unexpected column detection
    - Schema and type validation
    - Range checks
    - Missing value checks
    - Missing value handling
    - Report generation

    Args:
        df (pd.DataFrame): DataFrame to validate.
        schema (Dict): Schema definition for the data.
        logger: Logger instance to use.
        missing_strategy (str): Strategy for handling missing values ('drop', 'impute', 'keep').
        on_error (str): Error behavior on validation failure ('raise' or 'warn').

    Returns:
        pd.DataFrame: Cleaned DataFrame after applying missing value strategy.
    """
    logger.info("Starting data validation process.")
    report = {
        'missing_columns': [],
        'unexpected_columns': [],
        'type_mismatches': {},
        'missing_values': {}
    }

    check_unexpected_columns(df, schema, logger, on_error, report)
    check_schema_and_types(df, schema, logger, on_error, report)
    check_missing_values(df, schema, logger, report)
    df = handle_missing_values(df, missing_strategy, logger)
    save_validation_report(report, logger)
    return df
