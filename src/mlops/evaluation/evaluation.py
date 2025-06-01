"""Model evaluation module."""

import os
import pickle
import json
import logging
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
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

from mlops.features.features import (
    define_features_and_label,
    create_price_direction_label,
    prepare_features
)
from mlops.preproccess.preproccessing import split_data
from mlops.data_validation.data_validation import load_config

logger = logging.getLogger(__name__)
config = load_config("config.yaml")


class ModelEvaluator:
    """Handle model evaluation for both regression and classification tasks."""

    def __init__(self):
        """Initialize ModelEvaluator with config parameters."""
        self.config = config
        self.preprocessing_pipeline = None
        self.ensure_output_directories()
        self._load_preprocessing_pipeline()

    def ensure_output_directories(self) -> None:
        """Create necessary output directories."""
        os.makedirs("reports", exist_ok=True)
        os.makedirs("plots", exist_ok=True)

    def _load_preprocessing_pipeline(self) -> None:
        """Load preprocessing artifacts (scaler, feature selections)."""
        artifacts_config = self.config.get("artifacts", {})
        pipeline_path = artifacts_config.get(
            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
        )

        if os.path.exists(pipeline_path):
            with open(pipeline_path, 'rb') as f:
                self.preprocessing_pipeline = pickle.load(f)
            logger.info(
                f"Preprocessing pipeline loaded from {pipeline_path}")
        else:
            logger.warning(
                f"Preprocessing pipeline not found at {pipeline_path}")

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
            "save_path", "models/linear_regression.pkl"
        )
        direction_model_path = model_config.get("logistic_regression", {}).get(
            "save_path", "models/logistic_regression.pkl"
        )

        price_model = self.load_model(price_model_path)
        direction_model = self.load_model(direction_model_path)

        return price_model, direction_model

    def prepare_test_data(self, df: pd.DataFrame) -> Tuple[
            pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare test data with same preprocessing as training.

        Args:
            df: Full DataFrame

        Returns:
            tuple: (X_test_reg, X_test_class,
                     y_test_regression, y_test_classification)
        """
        # Recreate the same preprocessing steps as training
        feature_cols, label_col = define_features_and_label()
        df_with_direction = create_price_direction_label(df, label_col)
        X, y_regression, y_classification = prepare_features(
            df_with_direction, feature_cols, label_col
        )

        # Split data (using same random state as training)
        _, X_test, _, y_test_reg = split_data(X, y_regression)
        _, X_test_class, _, y_test_class = split_data(X, y_classification)

        if self.preprocessing_pipeline is None:
            logger.warning(
                "No preprocessing pipeline found. Using raw features.")
            return X_test, X_test_class, y_test_reg, y_test_class

        # Apply scaling
        scaler = self.preprocessing_pipeline['scaler']
        X_test_scaled = scaler.transform(X_test)
        X_test_class_scaled = scaler.transform(X_test_class)

        # Apply feature selection
        pipeline = self.preprocessing_pipeline
        selected_features_reg = pipeline['selected_features_reg']
        selected_features_class = pipeline['selected_features_class']
        all_feature_cols = self.preprocessing_pipeline['all_feature_cols']

        # Get feature indices
        feature_indices_reg = [
            all_feature_cols.index(col) for col in selected_features_reg
        ]
        feature_indices_class = [
            all_feature_cols.index(col) for col in selected_features_class
        ]

        X_test_reg_final = X_test_scaled[:, feature_indices_reg]
        X_test_class_final = X_test_class_scaled[:, feature_indices_class]

        shape_msg = (f"Test data prepared - Reg: {X_test_reg_final.shape}, "
                     f"Class: {X_test_class_final.shape}")
        logger.info(shape_msg)

        return X_test_reg_final, X_test_class_final, y_test_reg, y_test_class

    def evaluate_regression_model(self, model: Any, X_test: np.ndarray,
                                  y_test: pd.Series,
                                  df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate regression model and generate visualizations.

        Args:
            model: Trained regression model
            X_test: Test features
            y_test: Test regression targets
            df: Full DataFrame for plotting

        Returns:
            Dictionary of regression metrics
        """
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5  # Calculate RMSE manually for compatibility

        logger.info(f"Linear Regression Test RMSE: {rmse:.4f}")

        # Create actual vs predicted plot
        self.plot_regression_predictions(df, y_test, predictions)

        metrics = {"RMSE": rmse}
        return metrics

    def evaluate_classification_model(self, model: Any, X_test: np.ndarray,
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
        probabilities = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, 'predict_proba') else None
        )

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        try:
            if probabilities is not None:
                roc_auc = roc_auc_score(y_test, probabilities)
            else:
                roc_auc = roc_auc_score(y_test, predictions)
        except ValueError:
            # Handle case where only one class is present
            roc_auc = 0.5
            warning_msg = ("ROC AUC could not be calculated - "
                           "only one class present in test set")
            logger.warning(warning_msg)

        logger.info(f"Logistic Regression Test Accuracy: {accuracy:.4f}")
        logger.info(f"Logistic Regression Test F1 Score: {f1:.4f}")
        logger.info(f"Logistic Regression Test ROC AUC: {roc_auc:.4f}")

        # Generate confusion matrix plot
        self.plot_confusion_matrix(y_test, predictions)

        # Generate classification report
        class_report = classification_report(
            y_test, predictions, output_dict=True)

        metrics = {
            "Accuracy": accuracy,
            "F1 Score": f1,
            "ROC AUC": roc_auc,
            "Confusion Matrix": confusion_matrix(y_test, predictions).tolist(),
            "Classification Report": class_report
        }

        return metrics

    def plot_confusion_matrix(self, y_test: pd.Series,
                              predictions: np.ndarray) -> None:
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

    def plot_regression_predictions(self, df: pd.DataFrame, y_true: pd.Series,
                                    y_pred: np.ndarray,
                                    timestamp_col: str = "timestamp") -> None:
        """
        Plot actual vs predicted prices over time.

        Args:
            df: Full DataFrame with timestamp
            y_true: True prices
            y_pred: Predicted prices
            timestamp_col: Name of timestamp column
        """
        # Create a DataFrame for plotting
        # Note: This assumes the test set corresponds to the last portion
        if timestamp_col in df.columns:
            # Get timestamps for the test set (last y_true.shape[0] entries)
            timestamps = df[timestamp_col].iloc[-len(y_true):].values
        else:
            # Create dummy timestamps if timestamp column not available
            timestamps = range(len(y_true))

        df_plot = pd.DataFrame({
            timestamp_col: timestamps,
            "actual": y_true.values,
            "predicted": y_pred
        })

        plt.figure(figsize=(15, 8))
        plt.plot(df_plot[timestamp_col], df_plot["actual"],
                 label="Actual BTC Price", marker='o', markersize=3, alpha=0.7)
        plt.plot(df_plot[timestamp_col], df_plot["predicted"],
                 label="Predicted BTC Price", marker='x',
                 markersize=3, alpha=0.7)
        plt.xlabel("Timestamp")
        plt.ylabel("BTC Price (USDT)")
        plt.title("Actual vs Predicted BTC Prices Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        save_path = "plots/price_prediction_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(
            "Price prediction plot saved to plots/price_prediction_plot.png"
            )

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
        metrics_path = artifacts_config.get(
            "metrics_path", "reports/evaluation_metrics.json"
        )

        with open(metrics_path, 'w') as f:
            json.dump(metrics_report, f, indent=2, default=str)

        logger.info(f"Metrics report saved to {metrics_path}")

    def evaluate_all_models(self, df: pd.DataFrame) -> Tuple[
            Dict[str, float], Dict[str, Any]]:
        """
        Evaluate both models and generate all reports.

        Args:
            df: Full DataFrame for evaluation

        Returns:
            tuple: (regression_metrics, classification_metrics)
        """
        # Load models
        price_model, direction_model = self.load_both_models()

        # Prepare test data with preprocessing
        X_test_reg, X_test_class, y_test_reg, y_test_class = (
            self.prepare_test_data(df)
        )

        # Evaluate regression model
        logger.info("Evaluating regression model...")
        regression_metrics = self.evaluate_regression_model(
            price_model, X_test_reg, y_test_reg, df
        )

        # Evaluate classification model
        logger.info("Evaluating classification model...")
        classification_metrics = self.evaluate_classification_model(
            direction_model, X_test_class, y_test_class
        )

        # Save metrics report
        self.save_metrics_report(regression_metrics, classification_metrics)

        return regression_metrics, classification_metrics


def evaluate_models(df: pd.DataFrame = None) -> Tuple[
        Dict[str, float], Dict[str, Any]]:
    """
    Main function to evaluate both models.

    Args:
        df: DataFrame for evaluation.
            If None, loads processed data from config path

    Returns:
        tuple: (regression_metrics, classification_metrics)
    """
    if df is None:
        # Load processed data
        processed_path = config.get("data_source", {}).get(
            "processed_path", "./data/processed/processed_data.csv"
        )
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
            logger.info(f"Loaded processed data from {processed_path}")
        else:
            raise FileNotFoundError(
                f"Processed data not found at {processed_path}")

    evaluator = ModelEvaluator()
    return evaluator.evaluate_all_models(df)


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
