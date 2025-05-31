import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List
from data_validation.data_validation import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    config = load_config("config.yaml")
    params = config.get("preprocessing", {})
    data_split = config.get("data_split", {})
except Exception as e:
    logger.error(f"Failed to load config file: {e}")
    raise


def split_data(X, y):
    """
    Splits data into training and testing sets.

    Args:
        X: feature matrix
        y: target variable
        test_size: float, proportion of dataset to include in the test split

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    test_size = config.get("data_split", {}).get("test_size")
    random_state = config.get("data_split", {}).get("random_state")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(df: pd.DataFrame, selected_cols: List[str]) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Scales the selected features using StandardScaler.

    Args:
        df: Input DataFrame.
        selected_cols: List of column names to scale.

    Returns:
        Scaled feature matrix and the fitted scaler.
    """
    try:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df[selected_cols])
        X_test = scaler.transform(df[selected_cols])
        logger.info(f"Successfully scaled features: {selected_cols}")
        return X_train, X_test, scaler
    except Exception as e:
        logger.error(f"Error in scale_features: {e}")
        raise

def smote_oversample(X, y) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Applies SMOTE oversampling if class imbalance ratio > 1.5.

    Args:
        X: Feature matrix.
        y: Target labels.

    Returns:
        Tuple of resampled X and y.
    """
    try:
        class_counts = pd.Series(y).value_counts().to_dict()
        maj = max(class_counts, key=class_counts.get)
        min_ = min(class_counts, key=class_counts.get)
        ratio = class_counts[maj] / class_counts[min_]

        logger.info(f"Class distribution: {class_counts}")

        if ratio > 1.5:
            logger.info("Applying SMOTE oversampling...")
            sm = SMOTE(sampling_strategy='auto', random_state=params.get("random_state", 42))
            X_res, y_res = sm.fit_resample(X, y)
            logger.info("SMOTE applied successfully.")
        else:
            X_res, y_res = X, y
            logger.info("Class ratio below threshold. SMOTE not applied.")
        return X_res, y_res

    except Exception as e:
        logger.error(f"Error in smote_oversample: {e}")
        raise
