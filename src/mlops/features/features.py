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


import logging
import pandas as pd
<<<<<<< HEAD
from data_validation.data_validation import load_config
from sklearn.ensemble import RandomForestRegressor 
=======
from sklearn.ensemble import RandomForestRegressor
from src.mlops.data_validation.data_validation import load_config
>>>>>>> 04b66c6f97f6b7beda6b1502a056837cfd0b3e5b

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load configuration with error handling
try:
    config = load_config("config.yaml")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise


def define_features_and_label():
    """
    Defines the feature columns and target label.
    """
<<<<<<< HEAD
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
=======
    try:
        feature_cols = config.get("symbols", [])
        label_col = config.get("target")

        logger.info(f"Feature columns: {feature_cols}")
        logger.info(f"Label column: {label_col}")

        return feature_cols, label_col
    except Exception as e:
        logger.error(f"Error in define_features_and_label: {e}")
        raise
>>>>>>> 04b66c6f97f6b7beda6b1502a056837cfd0b3e5b


def create_price_direction_label(df, label_col):
    """
    Creates a binary price direction column based on price changes.
    """
    try:
        df = df.sort_values('timestamp').copy()
        df['prev_price'] = df[label_col].shift(1)
        df['price_direction'] = (df[label_col] > df['prev_price']).astype(int)
        df.dropna(inplace=True)
        logger.info(f"Created price direction column based on {label_col}.")
        return df
    except Exception as e:
        logger.error(f"Error in create_price_direction_label: {e}")
        raise


def prepare_features(df, feature_cols, label_col):
    """
    Prepares feature matrix and target variables.
    """
    try:
        X = df[feature_cols]
        y_reg = df[label_col]
        y_class = df['price_direction']
        logger.info(f"Prepared features: X={X.shape}, y_reg={y_reg.shape}, y_class={y_class.shape}")
        return X, y_reg, y_class
    except Exception as e:
        logger.error(f"Error in prepare_features: {e}")
        raise


def select_features(df: pd.DataFrame, feature_cols: list):
    """
    Selects top N features using RandomForest importance.
    """
    try:
        X = df[feature_cols]
        y = df[config.get("target")]

        rf_config = config.get("feature_engineering", {}).get("feature_selection", {})
        params = rf_config.get("params", {})
        top_n = rf_config.get("top_n", 5)

        n_estimators = params.get("n_estimators", 100)
        random_state = params.get("random_state", 42)

<<<<<<< HEAD
    print(f"n_estimators {n_estimators},random_state {random_state}")
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

        
    
    rf.fit(X, y)
=======
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X, y)
>>>>>>> 04b66c6f97f6b7beda6b1502a056837cfd0b3e5b

        importances = rf.feature_importances_
        ranked = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

        selected_cols = [col for col, _ in ranked[:top_n]]
        logger.info(f"Top {top_n} selected features: {selected_cols}")

        return selected_cols

    except Exception as e:
        logger.error(f"Error in select_features: {e}")
        raise