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
        dict: Parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def check_unexpected_columns(df: pd.DataFrame, schema: Dict, logger, on_error: str, report: Dict):
    """
    Identify and log unexpected columns not defined in the schema.

    Args:
        df (pd.DataFrame): Input DataFrame.
        schema (Dict): Expected schema definition.
        logger (Logger): Logger object.
        on_error (str): 'raise' to throw error, 'warn' to log warning.
        report (Dict): Dictionary to store validation issues.
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


def check_schema_and_types(df: pd.DataFrame, schema: Dict, logger, on_error: str, report: Dict):
    """
    Validate presence and type of each required column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        schema (Dict): Schema with expected data types.
        logger (Logger): Logger object.
        on_error (str): Behavior on validation error.
        report (Dict): Report dictionary to populate with findings.
    """
    for col, props in schema.items():
        if col not in df.columns:
            if props.get('required', True):
                report['missing_columns'].append(col)
                msg = f"Missing required column: {col}"
                if on_error == 'raise':
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
                if on_error == 'raise':
                    logger.error(msg)
                    raise
                else:
                    logger.warning(msg)
        elif expected_type.startswith('float') and not pd.api.types.is_float_dtype(df[col]):
            report['type_mismatches'][col] = {'expected': expected_type, 'actual': actual_type}
            msg = f"Type mismatch in column '{col}': expected {expected_type}, got {actual_type}"
            if on_error == 'raise':
                logger.error(msg)
                raise TypeError(msg)
            else:
                logger.warning(msg)


def check_missing_values(df: pd.DataFrame, schema: Dict, logger, report: Dict):
    """
    Check for missing values in DataFrame columns based on schema.

    Args:
        df (pd.DataFrame): Input data.
        schema (Dict): Expected schema.
        logger (Logger): Logger for warnings.
        report (Dict): Dictionary for recording missing value counts.
    """
    for col in schema.keys():
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                report['missing_values'][col] = int(missing_count)
                logger.warning(f"Column '{col}' has {missing_count} missing values.")


def handle_missing_values(df: pd.DataFrame, strategy: str, logger) -> pd.DataFrame:
    """
    Handle missing values according to defined strategy.

    Args:
        df (pd.DataFrame): DataFrame to clean.
        strategy (str): Strategy - 'drop', 'impute', or 'keep'.
        logger (Logger): Logger to report the strategy used.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'impute':
        return df.fillna(method='ffill').fillna(method='bfill')
    elif strategy == 'keep':
        return df
    else:
        logger.warning(f"Unknown missing_values_strategy: {strategy}. Proceeding without changes.")
        return df


def save_validation_report(report: Dict, logger):
    """
    Save the validation report to a JSON file.

    Args:
        report (Dict): Validation summary dictionary.
        logger (Logger): Logger to confirm file saved.
    """
    os.makedirs('reports', exist_ok=True)
    with open('reports/validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved to 'reports/validation_report.json'")


def validate_data(df: pd.DataFrame, schema: Dict, logger, missing_strategy='drop', on_error='raise') -> pd.DataFrame:
    """
    Orchestrates the validation process of a DataFrame.

    Args:
        df (pd.DataFrame): Input data to validate.
        schema (Dict): Expected schema definition.
        logger (Logger): Logging object passed from main.
        missing_strategy (str): Strategy for handling missing values.
        on_error (str): Behavior when encountering errors - 'raise' or 'warn'.

    Returns:
        pd.DataFrame: Validated (and possibly cleaned) DataFrame.
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


if __name__ == '__main__':
    # For testing/debugging only
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    df = pd.read_csv('data/raw/example.csv')
    config = load_config('config.yaml')
    validate_data(df, config['schema'], logger,
                  missing_strategy=config.get('missing_values_strategy', 'drop'),
                  on_error=config.get('on_error', 'raise'))