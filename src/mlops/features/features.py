# import pandas as pd


# def load_data(file_path):
#     """
#     Loads data from a CSV file and returns a pandas DataFrame.
    
#     Args:
#         file_path: str, path to the CSV file
        
#     Returns:
#         pandas DataFrame
#     """
#     df = pd.read_csv(file_path)
#     return df


# def define_features_and_label(df, symbols):
#     """
#     Defines the feature columns and target label for regression and classification tasks.
    
#     Args:
#         df: pandas DataFrame
#         symbols: list of symbols
        
#     Returns:
#         tuple: (feature_cols, label_col)
#     """
#     feature_cols = [f"{symbol}_price" for symbol in symbols[1:]] + \
#                    [f"{symbol}_funding_rate" for symbol in symbols]
#     label_col = "BTCUSDT_price"
#     return feature_cols, label_col
#     print(f"Feature columns: {feature_cols}")
#     print(f"Label column: {label_col}")


# def create_price_direction_label(df, label_col):
#     """
#     Creates a binary price direction column based on price changes.
    
#     Args:
#         df: pandas DataFrame
#         label_col: name of the price column
        
#     Returns:
#         pandas DataFrame with price direction column added
#     """
#     df = df.sort_values('timestamp').copy()
#     df['prev_price'] = df[label_col].shift(1)
#     df['price_direction'] = (df[label_col] > df['prev_price']).astype(int)
#     df = df.dropna()
#     return df
#     print(df)


# def prepare_features(df, feature_cols, label_col):
#     """
#     Prepares feature matrix and target variables for machine learning.
    
#     Args:
#         df: pandas DataFrame
#         feature_cols: list of feature column names
#         label_col: name of the label column
        
#     Returns:
#         tuple: (X, y_reg, y_class) where X is features, y_reg is regression target, y_class is classification target
#     """
#     X = df[feature_cols]
#     y_reg = df[label_col]
#     y_class = df['price_direction']
#     # return X, y_reg, y_class
#     print(f"Features shape: {X.shape}, Regression target shape: {y_reg.shape}, Classification target shape: {y_class.shape}")


# define_features_and_label(load_data("./data/processed/futures_data_processed.csv"), ["ETHUSDT_price", "XRPUSDT_price", "ADAUSDT_price", "SOLUSDT_price", "BNBUSDT_price", "BTCUSDT_funding_rate", "ETHUSDT_funding_rate", "XRPUSDT_funding_rate", "ADAUSDT_funding_rate", "SOLUSDT_funding_rate", "BNBUSDT_funding_rate"])
# create_price_direction_label(load_data("./data/processed/futures_data_processed.csv"), "BTCUSDT_price")
# prepare_features(load_data("./data/processed/futures_data_processed.csv"),
#                  ["ETHUSDT_price", "XRPUSDT_price", "ADAUSDT_price", "SOLUSDT_price", "BNBUSDT_price", "BTCUSDT_funding_rate", "ETHUSDT_funding_rate", "XRPUSDT_funding_rate", "ADAUSDT_funding_rate", "SOLUSDT_funding_rate", "BNBUSDT_funding_rate"],
#                  "BTCUSDT_price")


import pandas as pd
from src.mlops.data_validation.data_validation import load_config
from sklearn.ensemble import RandomForestRegressor 

config = load_config("config.yaml")


def define_features_and_label():
    """
    Defines the feature columns and target label
    for regression and classification tasks.

    Args:
        df: pandas DataFrame
        symbols: list of symbols

    Returns:
        tuple: (feature_cols, label_col)
    """
    feature_cols = config.get("symbols", [])  # Use the symbols list directly as feature columns
    label_col = config.get("target")
    print(f"Feature columns: {feature_cols}")
    print(f"Label column: {label_col}")
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
    df = df.sort_values('timestamp').copy()
    df['prev_price'] = df[label_col].shift(1)
    df['price_direction'] = (df[label_col] > df['prev_price']).astype(int)
    df = df.dropna()
    print(df)
    return df


def prepare_features(df, feature_cols, label_col):
    """
    Prepares feature matrix and target variables for machine learning.

    Args:
        df: pandas DataFrame
        feature_cols: list of feature column names
        label_col: name of the label column

    Returns:
        tuple: (X, y_reg, y_class) where X is features, y_reg is regression target, y_class is classification target
    """
    X = df[feature_cols]
    y_reg = df[label_col]
    y_class = df['price_direction']
    print(f"Features shape: {X.shape}, Regression target shape: {y_reg.shape}, Classification target shape: {y_class.shape}")
    return X, y_reg, y_class

def select_features(df: pd.DataFrame, feature_cols: list):
    """
    Performs RandomForest‐based feature selection, keeping the top_n most
    important columns. Returns a list of selected column names.
    """

    # Prepare the training set for importance:
    X = df[feature_cols].copy()
    y = df[config.get("target")]  # This is the regression label by default

    # Instantiate and fit the RacndomForest
    rf = RandomForestRegressor
        n_estimators=config.get("feature_engineering", {}).get('feature_selection', {}).get('params', {}).get("n_estimators")
        random_state=config.get("feature_engineering", {}).get('feature_selection', {}).get('params', {}).get("random_state")
    
    rf.fit(X, y)

    # Rank features by importance
    imp = rf.feature_importances_
    ranked = sorted(zip(feature_cols, imp), key=lambda x: x[1], reverse=True)

    # Keep only the top_n names
    selected_cols = [col for col, _ in ranked[:top_n]]
    print(f"[select_features] top_{top_n} selected: {selected_cols}")
    return selected_cols