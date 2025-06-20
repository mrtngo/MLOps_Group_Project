import pandas as pd
from src.mlops.data_validation.data_validation import load_config
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, List, Dict
import yaml
import logging


def define_features_and_label(config: Dict):
    """
    Defines the feature columns and target label
    for regression and classification tasks.

    Returns:
        tuple: (feature_cols, label_col)
    """
    symbols = config.get("symbols", [])

    feature_cols = [
        f"{symbol}_price" for symbol in symbols if symbol != "BTCUSDT"
    ] + [
        f"{symbol}_funding_rate" for symbol in symbols
    ]

    label_col = "BTCUSDT_price"

    print(f"[define_features_and_label] Features: {feature_cols}")
    print(f"[define_features_and_label] Label: {label_col}")

    return feature_cols, label_col


def create_price_direction_label(df, label_col):
    """
    Creates a binary price direction column based on price changes.

    Args:
        df: pandas DataFrame
        label_col: name of the price column

    Returns:
        pandas DataFrame with price direction column added
    """
    print(df.head())
    df = df.sort_values('timestamp').copy()
    df['prev_price'] = df[label_col].shift(1)
    df['price_direction'] = (df[label_col] > df['prev_price']).astype(int)
    df = df.dropna()
    shape_msg = (
        f"[create_price_direction_label] Created price direction "
        f"shape={df.shape}"
    )
    print(shape_msg)
    return df


def prepare_features(df, feature_cols, label_col):
    """
    Prepares feature matrix and target variables for machine learning.

    Args:
        df: pandas DataFrame
        feature_cols: list of feature column names
        label_col: name of the label column

    Returns:
        tuple: (X, y_reg, y_class) where X is features,
                y_reg is regression target,
                y_class is classification target
    """
    X = df[feature_cols]
    y_reg = df[label_col]
    y_class = df['price_direction']
    shape_msg = (f"Features shape: {X.shape}, "
                 f"Regression target shape: {y_reg.shape}, "
                 f"Classification target shape: {y_class.shape}")
    print(shape_msg)
    return X, y_reg, y_class


def select_features(df: pd.DataFrame, feature_cols: list[str], target_col: str, config: Dict) -> list[str]:
    """
    Selects features based on correlation with the target variable.

    Args:
        df: DataFrame containing features and the target.
        feature_cols: List of potential feature columns.
        target_col: The name of the target column.
        config: Configuration dictionary.

    Returns:
        List of selected feature names.
    """
    logger = logging.getLogger("FeatureSelection")
    selection_config = config.get("feature_engineering", {}).get("feature_selection", {})
    correlation_threshold = selection_config.get("correlation_threshold", 0.05)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame for feature selection.")

    # Calculate correlations
    correlations = df[feature_cols + [target_col]].corr()[target_col].abs()

    # Select features with correlation above the threshold
    selected = correlations[correlations > correlation_threshold].index.tolist()
    selected.remove(target_col)  # Remove the target itself

    logger.info(f"Selected {len(selected)} features based on correlation > {correlation_threshold} with '{target_col}'")
    return selected


def get_training_and_testing_data(config: Dict, df: pd.DataFrame = None):
    """
    Load or split training and testing data.

    Args:
        config: Configuration dictionary.
        df: Optional DataFrame to split. If None, loads from processed path.

    Returns:
        tuple: (df_training, df_testing)
    """
    if df is None:
        # Load processed data from config path
        processed_path = config.get("data_source", {}).get(
            "processed_path", "./data/processed/processed_data.csv"
        )
        try:
            df = pd.read_csv(processed_path)
            load_msg = (f"[get_training_and_testing_data] Loaded data from "
                        f"{processed_path} | shape={df.shape}")
            print(load_msg)
        except FileNotFoundError:
            no_data_msg = (f"[get_training_and_testing_data] Warning: "
                           f"No processed data found at {processed_path}")
            print(no_data_msg)
            return None, None

    # Split data into training and testing sets
    # Using config split ratios
    test_size = config.get("data_split", {}).get("test_size", 0.2)

    # Simple chronological split for time series data
    split_index = int(len(df) * (1 - test_size))
    df_training = df.iloc[:split_index].copy()
    df_testing = df.iloc[split_index:].copy()

    split_msg = (f"[get_training_and_testing_data], "
                 f"Training shape: {df_training.shape}, "
                 f"Testing shape: {df_testing.shape}")
    print(split_msg)

    return df_training, df_testing


if __name__ == "__main__":
    print("features.py - Use functions by importing them in other modules")
