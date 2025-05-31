"""Model evaluation module."""

import os
import pickle
import json
import logging
from typing import Tuple, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, 
    roc_auc_score, 
    confusion_matrix, 
    classification_report,
    accuracy_score,
    f1_score
)

from src.mlops.features.features import get_training_and_testing_data
from src.mlops.data_validation.data_validation import load_config

logger = logging.getLogger(__name__)
config = load_config("config.yaml")


class ModelEvaluator:
    """Handle model evaluation for both regression and classification tasks."""
    
    def __init__(self):
        """Initialize ModelEvaluator with config parameters."""
        self.config = config
        self.ensure_output_directories()
    
    def ensure_output_directories(self) -> None:
        """Create necessary output directories."""
        os.makedirs("reports", exist_ok=True)
        os.makedirs("plots", exist_ok=True)
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a pickled model.
        
        Args:
            model_path: Path to the pickled model file
            
        Returns:
            Loaded model object
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"Model loaded from {model_path}")
        return model
    
    def load_both_models(self) -> Tuple[Any, Any]:
        """
        Load both price and direction models.
        
        Returns:
            tuple: (price_model, direction_model)
        """
        model_config = self.config.get("model", {})
        
        price_model_path = model_config.get("linear_regression", {}).get(
            "save_path", "models/price_model.pkl"
        )
        direction_model_path = model_config.get("logistic_regression", {}).get(
            "save_path", "models/direction_model.pkl"
        )
        
        price_model = self.load_model(price_model_path)
        direction_model = self.load_model(direction_model_path)
        
        return price_model, direction_model
    
    def prepare_test_data(self, df_testing: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare test features and targets.
        
        Args:
            df_testing: Testing DataFrame
            
        Returns:
            tuple: (X_test, y_test_regression, y_test_classification)
        """
        feature_cols = self.config.get("features", [])
        target_col = self.config.get("target")
        
        X_test = df_testing[feature_cols]
        y_test_regression = df_testing[target_col]
        y_test_classification = df_testing['price_direction']
        
        return X_test, y_test_regression, y_test_classification
    
    def evaluate_regression_model(self, model: Any, X_test: pd.DataFrame, 
                                y_test: pd.Series, df_testing: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate regression model and generate visualizations.
        
        Args:
            model: Trained regression model
            X_test: Test features
            y_test: Test regression targets
            df_testing: Full test DataFrame for plotting
            
        Returns:
            Dictionary of regression metrics
        """
        predictions = model.predict(X_test)
        rmse = mean_squared_error(y_test, predictions, squared=False)
        
        logger.info(f"Linear Regression Test RMSE: {rmse:.4f}")
        
        # Create actual vs predicted plot
        self.plot_regression_predictions(df_testing, y_test, predictions)
        
        metrics = {"RMSE": rmse}
        return metrics
    
    def evaluate_classification_model(self, model: Any, X_test: pd.DataFrame, 
                                    y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate classification model and generate visualizations.
        
        Args:
            model: Trained classification model
            X_test: Test features
            y_test: Test classification targets
            
        Returns:
            Dictionary of classification metrics
        """
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probabilities) if probabilities is not None else roc_auc_score(y_test, predictions)
        
        logger.info(f"Logistic Regression Test Accuracy: {accuracy:.4f}")
        logger.info(f"Logistic Regression Test F1 Score: {f1:.4f}")
        logger.info(f"Logistic Regression Test ROC AUC: {roc_auc:.4f}")
        
        # Generate confusion matrix plot
        self.plot_confusion_matrix(y_test, predictions)
        
        # Generate classification report
        class_report = classification_report(y_test, predictions, output_dict=True)
        
        metrics = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "Confusion Matrix": confusion_matrix(y_test, predictions).tolist(),
            "Classification Report": class_report
        }
        
        return metrics
    
    def plot_confusion_matrix(self, y_test: pd.Series, predictions: pd.Series) -> None:
        """
        Plot and save confusion matrix.
        
        Args:
            y_test: True labels
            predictions: Predicted labels
        """
        cm = confusion_matrix(y_test, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.title("Confusion Matrix - Price Direction Prediction")
        plt.xlabel("Predicted Direction")
        plt.ylabel("Actual Direction")
        plt.tight_layout()
        plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Confusion matrix saved to plots/confusion_matrix.png")
    
    def plot_regression_predictions(self, df_testing: pd.DataFrame, y_true: pd.Series, 
                                  y_pred: pd.Series, timestamp_col: str = "timestamp") -> None:
        """
        Plot actual vs predicted prices over time.
        
        Args:
            df_testing: Test DataFrame with timestamp
            y_true: True prices
            y_pred: Predicted prices
            timestamp_col: Name of timestamp column
        """
        df_plot = df_testing[[timestamp_col]].copy()
        df_plot["actual"] = y_true.values
        df_plot["predicted"] = y_pred
        df_plot = df_plot.sort_values(by=timestamp_col)
        
        plt.figure(figsize=(15, 8))
        plt.plot(df_plot[timestamp_col], df_plot["actual"], 
                label="Actual BTC Price", marker='o', markersize=3, alpha=0.7)
        plt.plot(df_plot[timestamp_col], df_plot["predicted"], 
                label="Predicted BTC Price", marker='x', markersize=3, alpha=0.7)
        plt.xlabel("Timestamp")
        plt.ylabel("BTC Price (USDT)")
        plt.title("Actual vs Predicted BTC Prices Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("plots/price_prediction_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Price prediction plot saved to plots/price_prediction_plot.png")
    
    def save_metrics_report(self, regression_metrics: Dict[str, float], 
                          classification_metrics: Dict[str, Any]) -> None:
        """
        Save evaluation metrics to JSON file.
        
        Args:
            regression_metrics: Dictionary of regression metrics
            classification_metrics: Dictionary of classification metrics
        """
        metrics_report = {
            "linear_regression": regression_metrics,
            "logistic_regression": classification_metrics
        }
        
        artifacts_config = self.config.get("artifacts", {})
        metrics_path = artifacts_config.get("metrics_path", "reports/evaluation_metrics.json")
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_report, f, indent=2, default=str)
        
        logger.info(f"Metrics report saved to {metrics_path}")
    
    def evaluate_all_models(self, df_testing: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Evaluate both models and generate all reports.
        
        Args:
            df_testing: Testing DataFrame
            
        Returns:
            tuple: (regression_metrics, classification_metrics)
        """
        # Load models
        price_model, direction_model = self.load_both_models()
        
        # Prepare test data
        X_test, y_test_regression, y_test_classification = self.prepare_test_data(df_testing)
        
        # Evaluate regression model
        logger.info("Evaluating regression model...")
        regression_metrics = self.evaluate_regression_model(
            price_model, X_test, y_test_regression, df_testing
        )
        
        # Evaluate classification model
        logger.info("Evaluating classification model...")
        classification_metrics = self.evaluate_classification_model(
            direction_model, X_test, y_test_classification
        )
        
        # Save metrics report
        self.save_metrics_report(regression_metrics, classification_metrics)
        
        return regression_metrics, classification_metrics


def evaluate_models(df_testing: pd.DataFrame = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Main function to evaluate both models.
    
    Args:
        df_testing: Optional test DataFrame. If None, loads from features.py
        
    Returns:
        tuple: (regression_metrics, classification_metrics)
    """
    if df_testing is None:
        logger.info("Loading test data from features module...")
        _, df_testing = get_training_and_testing_data()
    
    evaluator = ModelEvaluator()
    return evaluator.evaluate_all_models(df_testing)


def generate_report(config: Dict[str, Any]) -> None:
    """
    Generate evaluation report (for compatibility with existing main.py).
    
    Args:
        config: Configuration dictionary
    """
    logger.info("Generating evaluation report...")
    evaluate_models()


if __name__ == "__main__":
    # For standalone execution
    evaluate_models()
