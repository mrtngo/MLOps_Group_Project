import argparse
import logging
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import mlflow
import wandb
from src.mlops.data_load.data_load import fetch_data, load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_data_load(start_date: str, end_date: str):
    """
    Executes the data loading step as a standalone script.
    Initializes MLflow and W&B, fetches data, and logs outputs.
    """
    logger.info("--- Starting Standalone Data Load Step ---")
    
    # Load configuration from the root of the project
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
        name="data_load-standalone",
        job_type="data-loading"
    )

    try:
        with mlflow.start_run(run_name="data_load") as mlrun:
            logger.info(f"Fetching data from {start_date} to {end_date}...")
            df = fetch_data(start_date=start_date, end_date=end_date)
            logger.info(f"Raw data loaded | shape={df.shape}")

            params_to_log = {
                "start_date": start_date,
                "end_date": end_date,
                "symbols": config.get("symbols", [])
            }
            mlflow.log_params(params_to_log)
            wandb.config.update(params_to_log)

            raw_data_path = config.get("data_source", {}).get("raw_path", "data/raw/raw_data.csv")
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            df.to_csv(raw_data_path, index=False)
            mlflow.log_artifact(raw_data_path, "raw-data")

            wandb.log({"raw_data_rows": df.shape[0], "raw_data_columns": df.shape[1]})
            artifact = wandb.Artifact('raw-data', type='dataset', description="Raw data fetched via API")
            artifact.add_file(raw_data_path)
            wandb.log_artifact(artifact)
            logger.info(f"Logged raw data artifact to MLflow and W&B: {raw_data_path}")

        logger.info("--- Data Load Step Completed Successfully ---")

    except Exception as e:
        logger.exception("Data load step failed")
        if wandb_run:
            wandb.log({"status": "failed", "error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data loading pipeline step.")
    parser.add_argument("--start-date", default="2023-01-01", help="Start date for fetching data (YYYY-MM-DD).")
    parser.add_argument("--end-date", default="2023-12-31", help="End date for fetching data (YYYY-MM-DD).")
    args = parser.parse_args()

    run_data_load(args.start_date, args.end_date) 