"""Preprocessing utilities for scaling, splitting, and SMOTE for ML tasks."""

from mlops.utils.logger import setup_logger
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logger = setup_logger(__name__)


def split_data(X, y, config: Dict):
    """
    Splits data into training and testing sets.

    Args:
        X: feature matrix
        y: target variable
        config: Configuration dictionary.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    test_size = config.get("data_split", {}).get("test_size", 0.2)
    random_state = config.get("data_split", {}).get("random_state", 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    shape_msg = (
        f"Data split completed - Train: {X_train.shape}, " f"Test: {X_test.shape}"
    )
    logger.info(shape_msg)
    return X_train, X_test, y_train, y_test


def scale_features(
    df: pd.DataFrame, selected_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Scales the selected features using StandardScaler.

    Note: This function now expects a DataFrame that
    represents the training set, and returns scaled training data,
    scaled test data placeholder, and the fitted scaler.
    For proper train/test scaling, use this in conjunction with split_data.

    Args:
        df: Training DataFrame with selected columns
        selected_cols: List of column names to scale

    Returns:
        Tuple of (X_train_scaled, X_test_scaled_placeholder, fitted_scaler)
    """
    try:
        scaler = StandardScaler()

        # Fit scaler on the provided DataFrame (should be training data)
        X_scaled = scaler.fit_transform(df[selected_cols])

        # Return scaled training data and the scaler for later use on test data
        # The second return value is a placeholder
        # real test data should be scaled separately
        X_test_placeholder = np.array([])  # Placeholder

        logger.info("Successfully scaled features: %s", selected_cols)
        return X_scaled, X_test_placeholder, scaler

    except Exception as e:
        logger.error(f"Error in scale_features: {e}")
        raise


def scale_test_data(
    X_test: pd.DataFrame, scaler: StandardScaler, selected_cols: List[str]
) -> np.ndarray:
    """
    Scale test data using a pre-fitted scaler.

    Args:
        X_test: Test DataFrame
        scaler: Pre-fitted StandardScaler
        selected_cols: List of column names to scale

    Returns:
        Scaled test features
    """
    try:
        X_test_scaled = scaler.transform(X_test[selected_cols])
        logger.info(f"Test data scaled successfully: {X_test_scaled.shape}")
        return X_test_scaled
    except Exception as e:
        logger.error(f"Error scaling test data: {e}")
        raise


def smote_oversample(X, y, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies SMOTE oversampling if class imbalance ratio > threshold.

    Args:
        X: Feature matrix
        y: Target labels
        config: Configuration dictionary.

    Returns:
        Tuple of resampled X and y
    """
    try:
        # Handle both pandas Series and numpy arrays
        if hasattr(y, "value_counts"):
            class_counts = y.value_counts().to_dict()
        else:
            unique, counts = np.unique(y, return_counts=True)
            class_counts = dict(zip(unique, counts))

        if len(class_counts) < 2:
            warning_msg = "Only one class found in target. SMOTE not applicable."
            logger.warning(warning_msg)
            return X, y

        min_val = min(class_counts.values())
        if min_val == 0:
            logger.warning("One class has zero samples. SMOTE not applicable.")
            return X, y

        maj = max(class_counts, key=class_counts.get)
        min_ = min(class_counts, key=class_counts.get)
        ratio = class_counts[maj] / class_counts[min_]

        logger.info("Class distribution: %s", class_counts)

        # Get threshold from config
        threshold = (
            config.get("preprocessing", {})
            .get("sampling", {})
            .get("threshold_ratio", 1.5)
        )

        if ratio > threshold:
            logger.info("Applying SMOTE oversampling...")

            # Get SMOTE parameters from config
            sampling_params = (
                config.get("preprocessing", {}).get("sampling", {}).get("params", {})
            )
            sampling_strategy = sampling_params.get("sampling_strategy", "auto")
            random_state = sampling_params.get("random_state", 42)

            sm = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
            X_res, y_res = sm.fit_resample(X, y)

            logger.info("SMOTE apply successful. New shape: %s", X_res.shape)ful. New shape: {X_res.shape}")
            new_dist = dict(zip(*np.unique(y_res, return_counts=True)))
            logger.info(f"New class distribution: {new_dist}")
        else:
            X_res, y_res = X, y
            ratio_msg = (
                f"Class ratio ({ratio:.2f}) below threshold ({threshold}). "
                f"SMOTE not applied."
            )
            logger.info(ratio_msg)

        return X_res, y_res

    except Exception as e:
        logger.error(f"Error in smote_oversample: {e}")
        raise


def preprocess_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    feature_cols: List[str],
    config: Dict,
    apply_smote: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Complete preprocessing pipeline: scaling and optional SMOTE.

    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        y_train: Training targets
        feature_cols: List of feature column names
        config: Configuration dictionary.
        apply_smote: Whether to apply SMOTE oversampling

    Returns:
        Tuple of (X_train_processed, X_test_processed, y_train_processed,
                 y_test_unchanged, scaler)
    """
    try:
        logger.info("Starting preprocessing pipeline...")

        # Scale features
        X_train_scaled, _, scaler = scale_features(X_train, feature_cols)
        X_test_scaled = scale_test_data(X_test, scaler, feature_cols)

        # Apply SMOTE if requested
        if apply_smote:
            X_train_final, y_train_final = smote_oversample(
                X_train_scaled, y_train, config
            )
        else:
            X_train_final, y_train_final = X_train_scaled, y_train

        logger.info("Preprocessing pipeline completed successfully")
        return X_train_final, X_test_scaled, y_train_final, y_train, scaler

    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {e}")
        raise


if __name__ == "__main__":
    info_msg = "preproccessing.py - Use functions by importing them in other modules"
    logger.info(info_msg)
