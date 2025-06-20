import argparse
import logging
import os
import sys
import pandas as pd
import joblib

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import mlflow
import wandb
from src.mlops.data_load.data_load import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_inference(model_artifact_path: str, inference_data_path: str):
    """
    Executes the model inference step.
    - Loads a trained model and new data.
    - Generates predictions.
    - Logs predictions as an artifact.
    """
    logger.info("--- Starting Standalone Model Inference Step ---")

    # Find the project root directory and load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..')
    config_path = os.path.join(project_root, 'conf', 'config.yaml')
    config = load_config(config_path)

    # Set MLflow and W&B experiment
    mlflow_config = config.get("mlflow_tracking", {})
    experiment_name = mlflow_config.get("experiment_name", "MLOps-Group-Project-Experiment")
    mlflow.set_experiment(experiment_name)
    
    wandb_config = config.get("wandb", {})
    wandb.init(
        project=wandb_config.get("project", "mlops-project"),
        name="model_inference-standalone",
        config=config
    )

    with mlflow.start_run(run_name="model_inference") as mlrun:
        logger.info(f"Loading model from: {model_artifact_path}")
        model = joblib.load(model_artifact_path)
        
        logger.info(f"Loading inference data from: {inference_data_path}")
        inference_df = pd.read_csv(inference_data_path)

        logger.info(f"Generating predictions for {len(inference_df)} records...")
        predictions = model.predict(inference_df)
        
        # Save predictions to a file
        predictions_df = pd.DataFrame(predictions, columns=["prediction"])
        output_path = os.path.join(project_root, "data", "predictions", "predictions.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        
        logger.info(f"Predictions saved to: {output_path}")
        
        # Log predictions artifact
        mlflow.log_artifact(output_path, artifact_path="predictions")
        wandb.log({"predictions_table": wandb.Table(dataframe=predictions_df.head(100))})
        
        logger.info("--- Model Inference Step Completed Successfully ---")
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model inference script.")
    parser.add_argument("--model-artifact", type=str, required=True, help="Path to the model artifact.")
    parser.add_argument("--inference-data", type=str, required=True, help="Path to the inference data CSV.")
    args = parser.parse_args()
    run_inference(args.model_artifact, args.inference_data) 