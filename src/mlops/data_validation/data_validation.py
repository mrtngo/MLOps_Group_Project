# import pandas as pd
# import yaml
# import json
# import os
# from typing import Dict


# def load_config(config_path: str) -> Dict:
#     """
#     Load configuration schema from a YAML file.

#     Args:
#         config_path (str): Path to the YAML configuration file.

#     Returns:
#         dict: Parsed configuration dictionary containing schema and settings.
#     """
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)


# def check_unexpected_columns(df: pd.DataFrame, schema: Dict, logger, on_error: str, report: Dict):
#     """
#     Check for unexpected columns that are not defined in the schema.

#     Args:
#         df (pd.DataFrame): The input DataFrame to validate.
#         schema (Dict): Schema definition mapping columns to properties.
#         logger: Logger object for logging warnings/errors.
#         on_error (str): Global behavior on error ('raise' or 'warn').
#         report (Dict): Dictionary to store validation results.
#     """
#     expected_columns = set(schema.keys())
#     actual_columns = set(df.columns)
#     unexpected_cols = actual_columns - expected_columns
#     if unexpected_cols:
#         report['unexpected_columns'] = list(unexpected_cols)
#         msg = f"Unexpected columns found: {unexpected_cols}"
#         if on_error == 'raise':
#             logger.error(msg)
#             raise ValueError(msg)
#         else:
#             logger.warning(msg)


# def check_value_ranges(df: pd.DataFrame, col: str, props: Dict, logger, on_error: str, report: Dict):
#     """
#     Validate whether the values in a given column fall within allowed min/max bounds.

#     Args:
#         df (pd.DataFrame): The input DataFrame.
#         col (str): The name of the column to check.
#         props (Dict): Properties of the column, including min/max.
#         logger: Logger object.
#         on_error (str): Global behavior on error ('raise' or 'warn').
#         report (Dict): Dictionary to store validation results.
#     """
#     if 'min' in props or 'max' in props:
#         out_of_range = df[col][
#             ((props.get('min') is not None) & (df[col] < props['min'])) |
#             ((props.get('max') is not None) & (df[col] > props['max']))
#         ]
#         if not out_of_range.empty:
#             report.setdefault('out_of_range', {})[col] = {
#                 'count': len(out_of_range),
#                 'min_allowed': props.get('min'),
#                 'max_allowed': props.get('max')
#             }
#             msg = f"Column '{col}' has {len(out_of_range)} values out of allowed range"
#             col_error = props.get('on_error', on_error)
#             if col_error == 'raise':
#                 logger.error(msg)
#                 raise ValueError(msg)
#             else:
#                 logger.warning(msg)


# def check_schema_and_types(df: pd.DataFrame, schema: Dict, logger, on_error: str, report: Dict):
#     """
#     Validate each column in the schema:
#     - Required columns exist
#     - Data type matches expected
#     - Values are within allowed ranges

#     Args:
#         df (pd.DataFrame): The input DataFrame.
#         schema (Dict): Dictionary defining expected schema.
#         logger: Logger object.
#         on_error (str): Global error handling ('raise' or 'warn').
#         report (Dict): Dictionary to update with validation results.
#     """
#     for col, props in schema.items():
#         col_error = props.get('on_error', on_error)

#         if col not in df.columns:
#             if props.get('required', True):
#                 report['missing_columns'].append(col)
#                 msg = f"Missing required column: {col}"
#                 if col_error == 'raise':
#                     logger.error(msg)
#                     raise ValueError(msg)
#                 else:
#                     logger.warning(msg)
#             continue

#         expected_type = props['dtype']
#         actual_type = str(df[col].dtype)
#         if expected_type == 'datetime64[ns]':
#             try:
#                 df[col] = pd.to_datetime(df[col])
#             except Exception as e:
#                 report['type_mismatches'][col] = {'expected': expected_type, 'actual': actual_type}
#                 msg = f"Failed to convert column '{col}' to datetime: {e}"
#                 if col_error == 'raise':
#                     logger.error(msg)
#                     raise
#                 else:
#                     logger.warning(msg)
#         elif expected_type.startswith('float') and not pd.api.types.is_float_dtype(df[col]):
#             report['type_mismatches'][col] = {'expected': expected_type, 'actual': actual_type}
#             msg = f"Type mismatch in column '{col}': expected {expected_type}, got {actual_type}"
#             if col_error == 'raise':
#                 logger.error(msg)
#                 raise TypeError(msg)
#             else:
#                 logger.warning(msg)

#         check_value_ranges(df, col, props, logger, on_error, report)


# def check_missing_values(df: pd.DataFrame, schema: Dict, logger, report: Dict):
#     """
#     Check and report the number of missing values for each column.

#     Args:
#         df (pd.DataFrame): The input DataFrame.
#         schema (Dict): Dictionary defining schema columns.
#         logger: Logger object.
#         report (Dict): Dictionary to update with missing value info.
#     """
#     for col in schema.keys():
#         if col in df.columns:
#             missing_count = df[col].isnull().sum()
#             if missing_count > 0:
#                 report['missing_values'][col] = int(missing_count)
#                 logger.warning(f"Column '{col}' has {missing_count} missing values.")


# def handle_missing_values(df: pd.DataFrame, strategy: str, logger) -> pd.DataFrame:
#     """
#     Handle missing values using a specific strategy.

#     Args:
#         df (pd.DataFrame): The input DataFrame.
#         strategy (str): Strategy to use ('drop', 'impute', 'keep').
#         logger: Logger object.

#     Returns:
#         pd.DataFrame: Cleaned or unchanged DataFrame depending on strategy.
#     """
#     if strategy == 'drop':
#         return df.dropna()
#     elif strategy == 'impute':
#         return df.ffill().bfill()
#     elif strategy == 'keep':
#         return df
#     else:
#         logger.warning(f"Unknown missing_values_strategy: {strategy}. Proceeding without changes.")
#         return df


# def save_validation_report(report: Dict, logger):
#     """
#     Save the validation report as a JSON file.

#     Args:
#         report (Dict): The report dictionary to save.
#         logger: Logger object.
#     """
#     os.makedirs('reports', exist_ok=True)
#     with open('reports/validation_report.json', 'w') as f:
#         json.dump(report, f, indent=2)
#     logger.info("Validation report saved to 'reports/validation_report.json'")


# def validate_data(df: pd.DataFrame, schema: Dict, logger, missing_strategy: str = 'drop', on_error: str = 'raise') -> pd.DataFrame:
#     """
#     Main entry point for data validation.

#     This function performs:
#     - Unexpected column detection
#     - Schema and type validation
#     - Range checks
#     - Missing value checks
#     - Missing value handling
#     - Report generation

#     Args:
#         df (pd.DataFrame): DataFrame to validate.
#         schema (Dict): Schema definition for the data.
#         logger: Logger instance to use.
#         missing_strategy (str): Strategy for handling missing values ('drop', 'impute', 'keep').
#         on_error (str): Error behavior on validation failure ('raise' or 'warn').

#     Returns:
#         pd.DataFrame: Cleaned DataFrame after applying missing value strategy.
#     """
#     logger.info("Starting data validation process.")
#     report = {
#         'missing_columns': [],
#         'unexpected_columns': [],
#         'type_mismatches': {},
#         'missing_values': {}
#     }

#     check_unexpected_columns(df, schema, logger, on_error, report)
#     check_schema_and_types(df, schema, logger, on_error, report)
#     check_missing_values(df, schema, logger, report)
#     df = handle_missing_values(df, missing_strategy, logger)
#     save_validation_report(report, logger)
#     return df


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


def load_config(config_path: str, logger: Optional[logging.Logger] = None) -> Dict:
    if logger is None:
        logger = setup_logging()
        
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        if not isinstance(config, dict):
            raise ValueError("Configuration file must contain a valid dictionary")
            
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


def check_unexpected_columns(df: pd.DataFrame, schema: Dict, logger: logging.Logger, 
                           on_error: str, report: Dict) -> None:
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
            msg = f"Found {len(unexpected_cols)} unexpected columns: {sorted(unexpected_cols)}"
            logger.info(msg)
            
            if on_error == 'raise':
                logger.error("Raising error due to unexpected columns")
                raise ValueError(f"Unexpected columns found: {unexpected_cols}")
            else:
                logger.warning("Continuing despite unexpected columns")
        else:
            logger.debug("No unexpected columns found")
            
    except Exception as e:
        logger.error(f"Error checking unexpected columns: {e}")
        raise


def check_value_ranges(df: pd.DataFrame, col: str, props: Dict, logger: logging.Logger, 
                      on_error: str, report: Dict) -> None:
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
                        'actual_min': float(df[col].min()) if not df[col].empty else None,
                        'actual_max': float(df[col].max()) if not df[col].empty else None
                    }
                    
                    msg = f"Column '{col}' has {len(out_of_range)} values out of range [{min_val}, {max_val}]"
                    logger.info(msg)
                    
                    col_error = props.get('on_error', on_error)
                    if col_error == 'raise':
                        logger.error(f"Raising error for out-of-range values in column '{col}'")
                        raise ValueError(msg)
                    else:
                        logger.warning(f"Continuing despite out-of-range values in column '{col}'")
                else:
                    logger.debug(f"All values in column '{col}' are within range")
                    
    except KeyError as e:
        logger.error(f"Column '{col}' not found in DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"Error checking value ranges for column '{col}': {e}")
        raise


def check_schema_and_types(df: pd.DataFrame, schema: Dict, logger: logging.Logger, 
                          on_error: str, report: Dict) -> pd.DataFrame:
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
                            logger.error(f"Raising error for missing required column '{col}'")
                            raise ValueError(msg)
                    else:
                        logger.info(f"Optional column '{col}' not found, skipping")
                    continue

                # Type validation and conversion
                expected_type = props.get('dtype')
                if expected_type:
                    actual_type = str(df_copy[col].dtype)
                    logger.debug(f"Column '{col}': expected {expected_type}, actual {actual_type}")
                    
                    if expected_type == 'datetime64[ns]':
                        try:
                            df_copy[col] = pd.to_datetime(df_copy[col])
                            logger.debug(f"Successfully converted column '{col}' to datetime")
                        except Exception as e:
                            report['type_mismatches'][col] = {
                                'expected': expected_type, 
                                'actual': actual_type,
                                'error': str(e)
                            }
                            msg = f"Failed to convert column '{col}' to datetime: {e}"
                            logger.warning(msg)
                            
                            if col_error == 'raise':
                                logger.error(f"Raising error for datetime conversion failure in column '{col}'")
                                raise
                                
                    elif expected_type.startswith('float') and not pd.api.types.is_float_dtype(df_copy[col]):
                        try:
                            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                            logger.debug(f"Successfully converted column '{col}' to float")
                        except Exception as e:
                            report['type_mismatches'][col] = {
                                'expected': expected_type, 
                                'actual': actual_type,
                                'error': str(e)
                            }
                            msg = f"Failed to convert column '{col}' to float: {e}"
                            logger.warning(msg)
                            
                            if col_error == 'raise':
                                logger.error(f"Raising error for float conversion failure in column '{col}'")
                                raise

                # Range validation
                check_value_ranges(df_copy, col, props, logger, on_error, report)
                
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


def check_missing_values(df: pd.DataFrame, schema: Dict, logger: logging.Logger, 
                        report: Dict) -> None:
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
                        logger.info(f"Column '{col}': {missing_count} missing values ({missing_pct:.1f}%)")
                        total_missing += missing_count
                except Exception as e:
                    logger.error(f"Error checking missing values for column '{col}': {e}")
                    continue
                    
        if total_missing == 0:
            logger.info("No missing values found")
        else:
            logger.info(f"Total missing values across all columns: {total_missing}")
            
    except Exception as e:
        logger.error(f"Error checking missing values: {e}")
        raise


def handle_missing_values(df: pd.DataFrame, strategy: str, logger: logging.Logger) -> pd.DataFrame:
    try:
        original_shape = df.shape
        logger.info(f"Handling missing values with strategy: '{strategy}'")
        
        if strategy == 'drop':
            result_df = df.dropna()
            logger.info(f"Dropped rows with missing values: {original_shape[0]} -> {result_df.shape[0]} rows")
            
        elif strategy == 'impute':
            result_df = df.copy()
            # Forward fill, then backward fill
            result_df = result_df.ffill().bfill()
            imputed_count = df.isnull().sum().sum() - result_df.isnull().sum().sum()
            logger.info(f"Imputed {imputed_count} missing values using forward/backward fill")
            
        elif strategy == 'keep':
            result_df = df.copy()
            logger.info("Keeping all missing values as-is")
            
        else:
            logger.warning(f"Unknown missing_values_strategy: '{strategy}'. Keeping data unchanged.")
            result_df = df.copy()
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error handling missing values with strategy '{strategy}': {e}")
        raise


def save_validation_report(report: Dict, logger: logging.Logger, 
                          output_path: str = 'reports/validation_report.json') -> None:

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Add summary statistics to report
        report['summary'] = {
            'total_issues': (
                len(report.get('missing_columns', [])) +
                len(report.get('unexpected_columns', [])) +
                len(report.get('type_mismatches', {})) +
                sum(report.get('missing_values', {}).values()) +
                sum(len(v) if isinstance(v, list) else 1 for v in report.get('out_of_range', {}).values())
            ),
            'validation_passed': len(report.get('missing_columns', [])) == 0 and 
                               len(report.get('type_mismatches', {})) == 0
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


def validate_data(df: pd.DataFrame, schema: Dict, logger: Optional[logging.Logger] = None, 
                 missing_strategy: str = 'drop', on_error: str = 'raise') -> pd.DataFrame:
    
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
            
        logger.info(f"Starting data validation for DataFrame with shape {df.shape}")
        logger.info(f"Validation settings: missing_strategy='{missing_strategy}', on_error='{on_error}'")
        
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
        df_processed = check_schema_and_types(df, schema, logger, on_error, report)
        check_missing_values(df_processed, schema, logger, report)
        
        # Handle missing values
        df_final = handle_missing_values(df_processed, missing_strategy, logger)
        
        # Save validation report
        save_validation_report(report, logger)
        
        # Log summary
        total_issues = (len(report['missing_columns']) + 
                       len(report['unexpected_columns']) + 
                       len(report['type_mismatches']))
        
        if total_issues == 0:
            logger.info("✓ Data validation completed successfully with no issues")
        else:
            logger.info(f"⚠ Data validation completed with {total_issues} issues (see report for details)")
            
        logger.info(f"Final DataFrame shape: {df_final.shape}")
        return df_final
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        raise