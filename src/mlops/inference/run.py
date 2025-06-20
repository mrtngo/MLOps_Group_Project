import argparse
import logging
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import mlflow
import wandb
from src.mlops.data_validation.data_validation import load_config
from src.mlops.inference.inference import ModelInferencer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_inference(input_csv: str, output_csv: str):
    """
    Executes the batch inference step.
    - Loads new data from an input CSV.
    - Loads the trained models and preprocessing pipeline.
    - Generates predictions.
    - Saves predictions to an output CSV and logs artifacts.
    """
    logger.info("--- Starting Standalone Batch Inference Step ---")

    config = load_config("conf/config.yaml")

    # Set MLflow experiment
    mlflow_config = config.get("mlflow_tracking", {})
    experiment_name = mlflow_config.get("experiment_name", "MLOps-Group-Project-Experiment")
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to '{experiment_name}'")

    # Initialize a new W&B run
    wandb_config = config.get("wandb", {})
    wandb_run = wandb.init(
        project=wandb_config.get("project", "mlops-project"),
        entity=wandb_config.get("entity"),
        name="inference-standalone",
        job_type="inference"
    )

    try:
        with mlflow.start_run(run_name="inference") as mlrun:
            # --- 1. Load Input Data ---
            logger.info(f"Loading new data for inference from: {input_csv}")
            if not os.path.exists(input_csv):
                logger.error(f"Input data file not found at {input_csv}")
                sys.exit(1)
            df_input = pd.read_csv(input_csv)
            mlflow.log_param("input_csv", input_csv)
            wandb.config.update({"input_csv": input_csv})

            # --- 2. Run Inference ---
            logger.info("Initializing ModelInferencer and generating predictions...")
            inferencer = ModelInferencer()
            
            # Use the correct prediction method
            predictions = inferencer.predict_both(df_input)
            
            # Combine predictions with input data
            df_predictions = df_input.copy()
            df_predictions['predicted_price'] = predictions['price_predictions']
            df_predictions['predicted_binned_price'] = predictions['direction_predictions']
            df_predictions['prediction_probability'] = predictions['direction_probabilities']

            logger.info(f"Inference complete. Shape of predictions: {df_predictions.shape}")

            # --- 3. Save and Log Output ---
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            df_predictions.to_csv(output_csv, index=False)
            logger.info(f"Predictions saved to: {output_csv}")

            # Log to MLflow and W&B
            mlflow.log_artifact(input_csv, "inference-input-data")
            mlflow.log_artifact(output_csv, "inference-predictions")

            # Log a sample of predictions to W&B
            prediction_sample = df_predictions.head(50)
            wandb.log({"prediction_samples": wandb.Table(dataframe=prediction_sample)})

            # --- 4. Log Visualizations ---
            logger.info("Generating and logging visualizations...")
            
            # Log prediction artifact to W&B
            prediction_artifact = wandb.Artifact("predictions", type="result")
            prediction_artifact.add_file(output_csv)
            wandb.log_artifact(prediction_artifact)
            
            logger.info("All inference artifacts logged successfully.")

        logger.info("--- Batch Inference Step Completed Successfully ---")

    except Exception as e:
        logger.exception("Inference step failed")
        if wandb_run:
            wandb.log({"status": "failed", "error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the batch inference pipeline step.")
    
    # You would typically provide a path to new, unseen data here
    parser.add_argument(
        "--input-csv", 
        default="data/raw/test.csv", # Using test.csv as an example
        help="Path to the input CSV file with new data for inference."
    )
    parser.add_argument(
        "--output-csv", 
        default="data/processed/predictions.csv",
        help="Path to save the output CSV file with predictions."
    )
    args = parser.parse_args()

    run_inference(args.input_csv, args.output_csv) 