"""Model training module."""

import os
import pickle
import logging
from typing import Tuple, Any

import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_auc_score

from src.mlops.features.features import define_features_and_label, create_price_direction_label, prepare_features, select_features
from src.mlops.data_validation.data_validation import load_config

logger = logging.getLogger(__name__)
config = load_config("config.yaml")


class ModelTrainer:
    """Handle model training for both regression and classification tasks."""
    
    def __init__(self):
        """Initialize ModelTrainer with config parameters."""
        self.config = config
        self.model_config = self.config.get("model", {})
        self.ensure_model_directory()
    
    def ensure_model_directory(self) -> None:
        """Create models directory if it doesn't exist."""
        os.makedirs("models", exist_ok=True)
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare features and targets from training data using features.py functions.
        
        Args:
            df: Training DataFrame with raw data
            
        Returns:
            tuple: (X, y_regression, y_classification)
        """
        # Get feature columns and label column from config
        feature_cols, label_col = define_features_and_label()
        
        # Create price direction labels
        df_with_direction = create_price_direction_label(df, label_col)
        
        # Check if feature selection is enabled
        feature_selection_config = self.config.get("feature_engineering", {}).get("feature_selection", {})
        if feature_selection_config.get("enabled", False):
            logger.info("Performing feature selection...")
            selected_features = select_features(df_with_direction, feature_cols)
            feature_cols = selected_features
        
        # Prepare final features and targets
        X, y_regression, y_classification = prepare_features(df_with_direction, feature_cols, label_col)
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Regression target shape: {y_regression.shape}")
        logger.info(f"Classification target shape: {y_classification.shape}")
        
        return X, y_regression, y_classification
    
    def train_linear_regression(self, X: pd.DataFrame, y: pd.Series) -> LinearRegression:
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
        rmse = mean_squared_error(y, predictions, squared=False)
        logger.info(f"Linear Regression Training RMSE: {rmse:.4f}")
        
        # Save model
        save_path = lr_config.get("save_path", "models/price_model.pkl")
        self._save_model(model, save_path)
        
        return model
    
    def train_logistic_regression(self, X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
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
        
        model = LogisticRegression(**params)
        model.fit(X, y)
        
        # Calculate training metrics for logging
        predictions = model.predict(X)
        roc_auc = roc_auc_score(y, predictions)
        logger.info(f"Logistic Regression Training ROC AUC: {roc_auc:.4f}")
        
        # Save model
        save_path = log_config.get("save_path", "models/direction_model.pkl")
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
    
    def train_all_models(self, df: pd.DataFrame) -> Tuple[LinearRegression, LogisticRegression]:
        """
        Train both regression and classification models.
        
        Args:
            df: Training DataFrame with raw data
            
        Returns:
            tuple: (price_model, direction_model)
        """
        X, y_regression, y_classification = self.prepare_data(df)
        
        logger.info("Training Linear Regression model...")
        price_model = self.train_linear_regression(X, y_regression)
        
        logger.info("Training Logistic Regression model...")
        direction_model = self.train_logistic_regression(X, y_classification)
        
        return price_model, direction_model


def get_training_and_testing_data():
    """
    Placeholder function to load training and testing data.
    This should be implemented based on your data loading requirements.
    
    Returns:
        tuple: (df_training, df_testing)
    """
    # This is a placeholder - implement based on your data loading logic
    logger.warning("get_training_and_testing_data() is not implemented. Please implement data loading logic.")
    return None, None


def train_model(df: pd.DataFrame = None) -> Tuple[LinearRegression, LogisticRegression]:
    """
    Main function to train both models.
    
    Args:
        df: Optional preprocessed DataFrame. If None, loads from get_training_and_testing_data
        
    Returns:
        tuple: (price_model, direction_model)
    """
    if df is None:
        logger.info("Loading training data...")
        df_training, _ = get_training_and_testing_data()
        if df_training is None:
            raise ValueError("No training data available. Please provide a DataFrame or implement get_training_and_testing_data()")
    else:
        df_training = df
    
    trainer = ModelTrainer()
    return trainer.train_all_models(df_training)


if __name__ == "__main__":
    # For standalone execution
    train_model()