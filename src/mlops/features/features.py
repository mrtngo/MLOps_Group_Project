import pandas as pd
from mlops.data_validation.data_validation import load_config
from sklearn.ensemble import RandomForestRegressor

config = load_config("conf/config.yaml")


def define_features_and_label():
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


def select_features(df: pd.DataFrame, feature_cols: list):
    """
    Performs RandomForest‐based feature selection, keeping the top_n most
    important columns. Returns a list of selected column names.
    """

    # Prepare the training set for importance:
    X = df[feature_cols].copy()
    y = df[config.get("target")]  # This is the regression label by default

    # Instantiate and fit the RandomForest
    feature_selection_config = config.get("feature_engineering", {}).get(
        'feature_selection', {}
    )
    params_config = feature_selection_config.get('params', {})
    n_estimators = params_config.get("n_estimators")
    random_state = params_config.get("random_state")

    print(f"n_estimators {n_estimators}, random_state {random_state}")
    rf = RandomForestRegressor(
        n_estimators=n_estimators, random_state=random_state
    )

    rf.fit(X, y)

    # Rank features by importance
    imp = rf.feature_importances_
    ranked = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)

    # Keep only the top_n names
    top_n = feature_selection_config.get('top_n')
    selected_cols = [col for col, _ in ranked[:top_n]]
    select_msg = f"[select_features] top_{top_n} selected: {selected_cols}"
    print(select_msg)
    return selected_cols


def get_training_and_testing_data(df: pd.DataFrame = None):
    """
    Load or split training and testing data.

    Args:
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
