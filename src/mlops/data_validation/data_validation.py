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
    Check whether values in a column are within defined min and max bounds.
    If 'on_error' is defined at the column level, it overrides the global behavior.
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
    Validate presence, data type, and allowed range for each column in the schema.
    Allows per-column 'on_error' override.
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
    Identify missing values in each schema-defined column and update the report.
    """
    for col in schema.keys():
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                report['missing_values'][col] = int(missing_count)
                logger.warning(f"Column '{col}' has {missing_count} missing values.")


def handle_missing_values(df: pd.DataFrame, strategy: str, logger) -> pd.DataFrame:
    """
    Clean DataFrame from missing values using specified strategy.
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
    Persist the validation report to a file in JSON format.
    """
    os.makedirs('reports', exist_ok=True)
    with open('reports/validation_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    logger.info("Validation report saved to 'reports/validation_report.json'")


def validate_data(df: pd.DataFrame, schema: Dict, logger, missing_strategy='drop', on_error='raise') -> pd.DataFrame:
    """
    Orchestrates the full validation pipeline for the input data.
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
