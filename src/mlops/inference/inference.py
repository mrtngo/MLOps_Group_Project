import os
import pickle
import logging
from typing import Tuple, Dict, Any, Union

import pandas as pd
import numpy as np

from mlops.features.features import define_features_and_label
from mlops.data_validation.data_validation import load_config

logger = logging.getLogger(__name__)
config = load_config("conf/config.yaml")


class ModelInferencer:
    """Handle model inference for both price and direction prediction."""

    def __init__(self):
        """Initialize ModelInferencer, load models+preprocessing pipeline."""
        self.config = config
        self.price_model = None
        self.direction_model = None
        self.preprocessing_pipeline = None
        self._load_models()
        self._load_preprocessing_pipeline()

    def _load_models(self) -> None:
        """Load both pickled models."""
        model_config = self.config.get("model", {})

        price_model_path = model_config.get("linear_regression", {}).get(
            "save_path", "models/linear_regression.pkl"
        )
        direction_model_path = model_config.get("logistic_regression", {}).get(
            "save_path", "models/logistic_regression.pkl"
        )

        self.price_model = self._load_single_model(price_model_path)
        self.direction_model = self._load_single_model(direction_model_path)

        logger.info("Both models loaded successfully for inference")

    def _load_preprocessing_pipeline(self) -> None:
        """Load preprocessing artifacts (scaler, feature selections)."""
        artifacts_config = self.config.get("artifacts", {})
        pipeline_path = artifacts_config.get(
            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
        )

        if os.path.exists(pipeline_path):
            with open(pipeline_path, 'rb') as f:
                self.preprocessing_pipeline = pickle.load(f)
            logger.info(f"Preprocessing pipeline loaded from {pipeline_path}")
        else:
            logger.warning(
                f"Preprocessing pipeline not found at {pipeline_path}")
            warning_msg = "Inference uses raw features without preprocessing"
            logger.warning(warning_msg)

    def _load_single_model(self, model_path: str) -> Any:
        """
        Load a single pickled model.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded model object

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Model loaded from {model_path}")
        return model

    def _validate_and_preprocess_input(self, df: pd.DataFrame) -> Tuple[
            np.ndarray, np.ndarray]:
        """
        Validate and preprocess input DataFrame for inference.

        Args:
            df: Input DataFrame

        Returns:
            tuple: (processed_features_reg, processed_features_class)

        Raises:
            ValueError: If required features are missing or preprocessing fails
        """
        # Get expected features from config
        feature_cols, _ = define_features_and_label()

        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        # Check for required features
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Select only the required features in the correct order
        df_features = df[feature_cols].copy()

        # Check for missing values
        if df_features.isnull().any().any():
            missing_msg = (
                "Input data contains missing values. Consider preprocessing.")
            logger.warning(missing_msg)
            # For inference, we'll forward fill then backward fill
            df_features = df_features.ffill().bfill()

        if self.preprocessing_pipeline is None:
            no_pipeline_msg = (
                "No preprocessing pipeline available. Using raw features.")
            logger.warning(no_pipeline_msg)
            return df_features.values, df_features.values

        # Apply scaling
        scaler = self.preprocessing_pipeline['scaler']
        features_scaled = scaler.transform(df_features)

        # Apply feature selection for each model
        selected_features_reg = (
            self.preprocessing_pipeline['selected_features_reg'])
        selected_features_class = (
            self.preprocessing_pipeline['selected_features_class'])
        all_feature_cols = (
            self.preprocessing_pipeline['all_feature_cols'])

        # Get feature indices
        feature_indices_reg = [
            all_feature_cols.index(col) for col in selected_features_reg
        ]
        feature_indices_class = [
            all_feature_cols.index(col) for col in selected_features_class
        ]

        features_reg = features_scaled[:, feature_indices_reg]
        features_class = features_scaled[:, feature_indices_class]

        shape_msg = (f"Input preprocessed. "
                     f"Reg features: {features_reg.shape}, "
                     f"Class features: {features_class.shape}")
        logger.info(shape_msg)

        return features_reg, features_class

    def predict_price(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict prices using the linear regression model.

        Args:
            df: DataFrame with required features

        Returns:
            Array of predicted prices
        """
        if self.price_model is None:
            raise RuntimeError("Price model not loaded")

        features_reg, _ = self._validate_and_preprocess_input(df)
        predictions = self.price_model.predict(features_reg)

        logger.info(f"Generated {len(predictions)} price predictions")
        return predictions

    def predict_direction(self, df: pd.DataFrame) -> Tuple[
            np.ndarray, np.ndarray]:
        """
        Predict price directions using the logistic regression model.

        Args:
            df: DataFrame with required features

        Returns:
            tuple: (predicted_directions, prediction_probabilities)
        """
        if self.direction_model is None:
            raise RuntimeError("Direction model not loaded")

        _, features_class = self._validate_and_preprocess_input(df)

        # Get class predictions
        direction_predictions = self.direction_model.predict(features_class)

        # Get prediction probabilities if available
        if hasattr(self.direction_model, 'predict_proba'):
            probabilities = (
                self.direction_model.predict_proba(features_class)[:, 1])
        else:
            # If no probabilities available, use decision function/predictions
            if hasattr(self.direction_model, 'decision_function'):
                probabilities = (
                    self.direction_model.decision_function(features_class))
            else:
                probabilities = direction_predictions.astype(float)

        logger.info(
            f"Generated {len(direction_predictions)} direction predictions")
        return direction_predictions, probabilities

    def predict_both(self, df: pd.DataFrame) -> Dict[
            str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Predict both price and direction for the input data.

        Args:
            df: DataFrame with required features

        Returns:
            Dictionary containing both predictions:
            {
                'price_predictions': np.ndarray,
                'direction_predictions': np.ndarray,
                'direction_probabilities': np.ndarray
            }
        """
        price_predictions = self.predict_price(df)
        direction_predictions, direction_probabilities = (
            self.predict_direction(df)
            )

        results = {
            'price_predictions': price_predictions,
            'direction_predictions': direction_predictions,
            'direction_probabilities': direction_probabilities
        }

        logger.info("Generated both price and direction predictions")
        return results


# Convenience functions for easy integration
def load_models() -> ModelInferencer:
    """
    Load and return a ModelInferencer instance.

    Returns:
        Initialized ModelInferencer with loaded models
    """
    return ModelInferencer()


def predict_price(df: pd.DataFrame,
                  inferencer: ModelInferencer = None) -> np.ndarray:
    """
    Predict prices for the given DataFrame.

    Args:
        df: DataFrame with required features
        inferencer: Optional pre-loaded ModelInferencer instance

    Returns:
        Array of predicted prices
    """
    if inferencer is None:
        inferencer = ModelInferencer()

    return inferencer.predict_price(df)


def predict_direction(df: pd.DataFrame,
                      inferencer: ModelInferencer = None) -> Tuple[
                          np.ndarray, np.ndarray]:
    """
    Predict price directions for the given DataFrame.

    Args:
        df: DataFrame with required features
        inferencer: Optional pre-loaded ModelInferencer instance

    Returns:
        tuple: (predicted_directions, prediction_probabilities)
    """
    if inferencer is None:
        inferencer = ModelInferencer()

    return inferencer.predict_direction(df)


def predict_both(df: pd.DataFrame,
                 inferencer: ModelInferencer = None) -> Dict[
                     str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Predict both price and direction for the given DataFrame.

    Args:
        df: DataFrame with required features
        inferencer: Optional pre-loaded ModelInferencer instance

    Returns:
        Dictionary containing both predictions
    """
    if inferencer is None:
        inferencer = ModelInferencer()

    return inferencer.predict_both(df)


def run_inference(df: pd.DataFrame, config_path: str, output_csv: str) -> None:
    """
    Run inference on a CSV file and save results.

    Args:
        df: Input DataFrame
        config_path: Path to configuration file (for compatibility, not used)
        output_csv: Path to save output CSV file
    """
    logger.info(f"shape: {df.shape}")

    # Load models and run inference
    inferencer = ModelInferencer()
    results = inferencer.predict_both(df)

    # Prepare output DataFrame
    output_df = df.copy()
    output_df['predicted_price'] = results['price_predictions']
    output_df['predicted_direction'] = results['direction_predictions']
    output_df['direction_probability'] = results['direction_probabilities']

    # Save results
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Inference results saved to {output_csv}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) == 4:
        # CLI usage: python inference.py input.csv config.yaml output.csv
        run_inference(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        usage_msg = (
            "Usage: python inference.py <input_csv> <config_path> <output_csv>"
        )
        logger.info(usage_msg)
        import_msg = "Or import and use the functions directly in your code"
        logger.info(import_msg)
