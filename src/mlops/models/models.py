"""Model training module."""

import os
import pickle
import logging
from typing import Tuple, Any

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score

from mlops.features.features import (
    define_features_and_label,
    create_price_direction_label,
    prepare_features,
    select_features
)
from mlops.preproccess.preproccessing import (
    split_data,
    scale_features,
    smote_oversample
)
from mlops.data_validation.data_validation import load_config

logger = logging.getLogger(__name__)
config = load_config("conf/config.yaml")


class ModelTrainer:
    """Handle model training for both regression and classification tasks."""

    def __init__(self):
        """Initialize ModelTrainer with config parameters."""
        self.config = config
        self.model_config = self.config.get("model", {})
        self.scaler = None
        self.selected_features_reg = None
        self.selected_features_class = None
        self.ensure_output_directories()

    def ensure_output_directories(self) -> None:
        """Create necessary output directories."""
        os.makedirs("models", exist_ok=True)
        artifacts_config = self.config.get("artifacts", {})
        pipeline_path = artifacts_config.get(
            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
        )
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[
            pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series,
            pd.Series]:
        """
        Prepare features and targets with train/test splits, scaling,
        and feature selection.

        Args:
            df: Training DataFrame with raw data

        Returns:
            tuple: (X_train_reg, X_train_class, y_train_reg, y_train_class,
                   y_test_reg, y_test_class)
        """
        # Get feature columns and label column from config
        feature_cols, label_col = define_features_and_label()

        # Create price direction labels
        df_with_direction = create_price_direction_label(df, label_col)

        # Prepare features and targets
        X, y_regression, y_classification = prepare_features(
            df_with_direction, feature_cols, label_col
        )

        # Split data for regression and classification
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = split_data(
            X, y_regression
        )
        X_train_class, X_test_class, y_train_class, y_test_class = split_data(
            X, y_classification
        )

        # Scale features - use regression training set to fit scaler
        X_train_reg_scaled, X_test_reg_scaled, self.scaler = scale_features(
            pd.DataFrame(X_train_reg, columns=feature_cols), feature_cols
        )

        # Apply same scaling to classification data
        X_train_class_scaled = self.scaler.transform(X_train_class)

        # Feature selection - create DataFrames for feature selection
        df_reg_train = pd.DataFrame(X_train_reg_scaled, columns=feature_cols)
        df_reg_train[self.config.get("target")] = y_train_reg.values
        df_reg_train['price_direction'] = y_train_class.values

        df_class_train = pd.DataFrame(
            X_train_class_scaled, columns=feature_cols
        )
        df_class_train[self.config.get("target")] = y_train_reg.values
        df_class_train['price_direction'] = y_train_class.values

        # Select features for each model
        self.selected_features_reg = select_features(
            df_reg_train, feature_cols
        )
        self.selected_features_class = select_features(
            df_class_train, feature_cols
        )

        # Apply feature selection
        feature_indices_reg = [
            feature_cols.index(col) for col in self.selected_features_reg
        ]
        feature_indices_class = [
            feature_cols.index(col) for col in self.selected_features_class
        ]

        X_train_reg_selected = X_train_reg_scaled[:, feature_indices_reg]
        X_train_class_selected = X_train_class_scaled[:, feature_indices_class]

        # Apply SMOTE to classification data if needed
        X_train_class_balanced, y_train_class_balanced = smote_oversample(
            X_train_class_selected, y_train_class
        )

        # Store test data for evaluation (will be scaled+selected during eval)
        self.X_test_reg = X_test_reg
        self.X_test_class = X_test_class
        self.y_test_reg = y_test_reg
        self.y_test_class = y_test_class
        self.feature_cols = feature_cols

        # Save preprocessing pipeline
        self._save_preprocessing_pipeline()

        logger.info(f"Regression features: {self.selected_features_reg}")
        logger.info(f"Classification features: {self.selected_features_class}")
        shape_msg = (
            f"Final training shapes - Reg: {X_train_reg_selected.shape}, "
            f"Class: {X_train_class_balanced.shape}"
        )
        logger.info(shape_msg)

        return (
            X_train_reg_selected,
            X_train_class_balanced,
            y_train_reg,
            y_train_class_balanced,
            y_test_reg,
            y_test_class
        )

    def _save_preprocessing_pipeline(self) -> None:
        """Save preprocessing artifacts (scaler, feature selections)."""
        artifacts_config = self.config.get("artifacts", {})
        pipeline_path = artifacts_config.get(
            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
        )

        preprocessing_pipeline = {
            'scaler': self.scaler,
            'selected_features_reg': self.selected_features_reg,
            'selected_features_class': self.selected_features_class,
            'all_feature_cols': self.feature_cols
        }

        with open(pipeline_path, 'wb') as f:
            pickle.dump(preprocessing_pipeline, f)

        logger.info(f"Preprocessing pipeline saved to {pipeline_path}")

    def train_linear_regression(self, X: pd.DataFrame,
                                y: pd.Series) -> LinearRegression:
        """
        Train linear regression model for price prediction.

        Args:
            X: Feature matrix
            y: Target values for regression

        Returns:
            Trained linear regression model
        """
        lr_config = self.model_config.get("linear_regression", {})
        params = lr_config.get("params", {})

        model = LinearRegression(**params)
        model.fit(X, y)

        # Calculate training RMSE for logging
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        rmse = mse ** 0.5  # Calculate RMSE manually for compatibility
        logger.info(f"Linear Regression Training RMSE: {rmse:.4f}")

        # Save model
        save_path = lr_config.get("save_path", "models/linear_regression.pkl")
        self._save_model(model, save_path)

        return model

    def train_logistic_regression(self, X: pd.DataFrame,
                                  y: pd.Series) -> LogisticRegression:
        """
        Train logistic regression model for direction prediction.

        Args:
            X: Feature matrix
            y: Target values for classification

        Returns:
            Trained logistic regression model
        """
        log_config = self.model_config.get("logistic_regression", {})
        params = log_config.get("params", {})

        # Fix penalty parameter if it's incorrectly specified in config
        if 'penalty' in params and params['penalty'] == '12':
            params['penalty'] = 'l2'
            logger.warning("Fixed penalty parameter from '12' to 'l2'")

        model = LogisticRegression(**params)
        model.fit(X, y)

        # Calculate training metrics for logging
        predictions = model.predict(X)
        try:
            roc_auc = roc_auc_score(y, predictions)
            logger.info(f"Logistic Regression Training ROC AUC: {roc_auc:.4f}")
        except ValueError:
            warning_msg = (
                "Could not calculate ROC AUC - "
                "possibly only one class in training data"
            )
            logger.warning(warning_msg)

        # Save model
        save_path = log_config.get(
            "save_path", "models/logistic_regression.pkl"
        )
        self._save_model(model, save_path)

        return model

    def _save_model(self, model: Any, save_path: str) -> None:
        """
        Save model to pickle file.

        Args:
            model: Trained model to save
            save_path: Path where to save the model
        """
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {save_path}")

    def train_all_models(self, df: pd.DataFrame) -> Tuple[
            LinearRegression, LogisticRegression]:
        """
        Train both regression and classification models.

        Args:
            df: Training DataFrame with raw data

        Returns:
            tuple: (price_model, direction_model)
        """
        # Prepare data with preprocessing
        data_prep = self.prepare_data(df)
        (X_train_reg, X_train_class, y_train_reg,
         y_train_class, y_test_reg, y_test_class) = data_prep

        logger.info("Training Linear Regression model...")
        price_model = self.train_linear_regression(X_train_reg, y_train_reg)

        logger.info("Training Logistic Regression model...")
        direction_model = self.train_logistic_regression(
            X_train_class, y_train_class
        )

        return price_model, direction_model


def get_training_and_testing_data():
    """
    Load training and testing data splits.

    Returns:
        tuple: (df_training, df_testing) - For now returns None,
               should be implemented
    """
    # This should load your actual train/test splits
    # For now, returning None to maintain compatibility
    warning_msg = "get_training_and_testing_data() is not fully implemented."
    logger.warning(warning_msg)
    return None, None


def train_model(df: pd.DataFrame) -> Tuple[
        LinearRegression, LogisticRegression]:
    """
    Main function to train both models.

    Args:
        df: Preprocessed DataFrame with all data

    Returns:
        tuple: (price_model, direction_model)
    """
    if df is None:
        raise ValueError("DataFrame is required for training")

    trainer = ModelTrainer()
    return trainer.train_all_models(df)


if __name__ == "__main__":
    # For standalone execution
    warning_msg = "Standalone execution requires a DataFrame input"
    logger.warning(warning_msg)
