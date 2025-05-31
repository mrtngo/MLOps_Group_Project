"""Model inference module."""

import os
import pickle
import logging
from typing import Tuple, Dict, Any, Union

import pandas as pd
import numpy as np

from src.mlops.data_validation.data_validation import load_config

logger = logging.getLogger(__name__)
config = load_config("config.yaml")


class ModelInferencer:
    """Handle model inference for both price and direction prediction."""
    
    def __init__(self):
        """Initialize ModelInferencer and load models."""
        self.config = config
        self.price_model = None
        self.direction_model = None
        self._load_models()
    
    def _load_models(self) -> None:
        """Load both pickled models."""
        model_config = self.config.get("model", {})
        
        price_model_path = model_config.get("linear_regression", {}).get(
            "save_path", "models/price_model.pkl"
        )
        direction_model_path = model_config.get("logistic_regression", {}).get(
            "save_path", "models/direction_model.pkl"
        )
        
        self.price_model = self._load_single_model(price_model_path)
        self.direction_model = self._load_single_model(direction_model_path)
        
        logger.info("Both models loaded successfully for inference")
    
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
    
    def _validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare input DataFrame for inference.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame with required features
            
        Raises:
            ValueError: If required features are missing
        """
        required_features = self.config.get("features", [])
        
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the required features in the correct order
        df_features = df[required_features].copy()
        
        # Check for missing values
        if df_features.isnull().any().any():
            logger.warning("Input data contains missing values. Consider preprocessing.")
        
        return df_features
    
    def predict_price(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict prices using the linear regression model.
        
        Args:
            df: Preprocessed DataFrame with required features
            
        Returns:
            Array of predicted prices
        """
        if self.price_model is None:
            raise RuntimeError("Price model not loaded")
        
        df_features = self._validate_input_data(df)
        predictions = self.price_model.predict(df_features)
        
        logger.info(f"Generated {len(predictions)} price predictions")
        return predictions
    
    def predict_direction(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict price directions using the logistic regression model.
        
        Args:
            df: Preprocessed DataFrame with required features
            
        Returns:
            tuple: (predicted_directions, prediction_probabilities)
        """
        if self.direction_model is None:
            raise RuntimeError("Direction model not loaded")
        
        df_features = self._validate_input_data(df)
        
        # Get class predictions
        direction_predictions = self.direction_model.predict(df_features)
        
        # Get prediction probabilities if available
        if hasattr(self.direction_model, 'predict_proba'):
            probabilities = self.direction_model.predict_proba(df_features)[:, 1]
        else:
            # If no probabilities available, use decision function or predictions
            if hasattr(self.direction_model, 'decision_function'):
                probabilities = self.direction_model.decision_function(df_features)
            else:
                probabilities = direction_predictions.astype(float)
        
        logger.info(f"Generated {len(direction_predictions)} direction predictions")
        return direction_predictions, probabilities
    
    def predict_both(self, df: pd.DataFrame) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Predict both price and direction for the input data.
        
        Args:
            df: Preprocessed DataFrame with required features
            
        Returns:
            Dictionary containing both predictions:
            {
                'price_predictions': np.ndarray,
                'direction_predictions': np.ndarray,
                'direction_probabilities': np.ndarray
            }
        """
        price_predictions = self.predict_price(df)
        direction_predictions, direction_probabilities = self.predict_direction(df)
        
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


def predict_price(df: pd.DataFrame, inferencer: ModelInferencer = None) -> np.ndarray:
    """
    Predict prices for the given DataFrame.
    
    Args:
        df: Preprocessed DataFrame with required features
        inferencer: Optional pre-loaded ModelInferencer instance
        
    Returns:
        Array of predicted prices
    """
    if inferencer is None:
        inferencer = ModelInferencer()
    
    return inferencer.predict_price(df)


def predict_direction(df: pd.DataFrame, inferencer: ModelInferencer = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict price directions for the given DataFrame.
    
    Args:
        df: Preprocessed DataFrame with required features
        inferencer: Optional pre-loaded ModelInferencer instance
        
    Returns:
        tuple: (predicted_directions, prediction_probabilities)
    """
    if inferencer is None:
        inferencer = ModelInferencer()
    
    return inferencer.predict_direction(df)


def predict_both(df: pd.DataFrame, inferencer: ModelInferencer = None) -> Dict[str, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """
    Predict both price and direction for the given DataFrame.
    
    Args:
        df: Preprocessed DataFrame with required features
        inferencer: Optional pre-loaded ModelInferencer instance
        
    Returns:
        Dictionary containing both predictions
    """
    if inferencer is None:
        inferencer = ModelInferencer()
    
    return inferencer.predict_both(df)


def run_inference(input_csv: str, config_path: str, output_csv: str) -> None:
    """
    Run inference on a CSV file and save results.
    
    Args:
        input_csv: Path to input CSV file
        config_path: Path to configuration file (unused, for compatibility)
        output_csv: Path to save output CSV file
    """
    # Load input data
    df = pd.read_csv(input_csv)
    logger.info(f"Loaded input data from {input_csv}, shape: {df.shape}")
    
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
        # Command line usage: python inference.py input.csv config.yaml output.csv
        run_inference(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        logger.info("Usage: python inference.py <input_csv> <config_path> <output_csv>")
        logger.info("Or import and use the functions directly in your code")
