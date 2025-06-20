import argparse
import logging
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import mlflow
import pandas as pd
import wandb
from src.mlops.data_validation.data_validation import load_config, validate_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_data_validation(input_artifact: str):
    """
    Executes the data validation step.
    - Loads data from an input artifact.
    - Validates it against a schema.
    - Logs summary stats and sample rows to W&B.
    - Logs the validated data as a new artifact.
    """
    logger.info("--- Starting Standalone Data Validation Step ---")

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
        name="data_validation-standalone",
        job_type="data-validation"
    )

    try:
        with mlflow.start_run(run_name="data_validation") as mlrun:
            # --- 1. Load Data ---
            logger.info(f"Loading raw data from: {input_artifact}")
            if not os.path.exists(input_artifact):
                logger.error(f"Input artifact not found at {input_artifact}. Please run the data_load step first.")
                sys.exit(1)
            df = pd.read_csv(input_artifact)
            mlflow.log_param("input_artifact", input_artifact)
            wandb.config.update({"input_artifact": input_artifact})

            # --- 2. Validate Data ---
            logger.info("Validating data against the schema...")
            schema_list = config.get("data_validation", {}).get("schema", {}).get("columns", [])
            schema = {col["name"]: col for col in schema_list}
            missing_strategy = config.get("data_validation", {}).get("missing_values_strategy", "drop")
            
            df_validated = validate_data(df, schema, logger, missing_strategy, on_error="warn")
            logger.info(f"Data validation completed. Shape after validation: {df_validated.shape}")

            # --- 3. Log W&B Tables (like the professor's screenshot) ---
            logger.info("Generating and logging summary statistics and sample rows to W&B...")
            
            # Create summary_stats table
            summary_stats = df_validated.describe().reset_index()
            summary_table = wandb.Table(dataframe=summary_stats)
            wandb.log({"summary_stats": summary_table})
            
            # Create sample_rows table
            sample_rows = df_validated.head(20)
            sample_table = wandb.Table(dataframe=sample_rows)
            wandb.log({"sample_rows": sample_table})

            logger.info("Successfully logged W&B Tables.")

            # --- 4. Log Output Artifact ---
            validated_data_path = config.get("data_source", {}).get("processed_path", "data/processed/validated_data.csv")
            os.makedirs(os.path.dirname(validated_data_path), exist_ok=True)
            df_validated.to_csv(validated_data_path, index=False)
            
            mlflow.log_artifact(validated_data_path, "validated-data")
            
            artifact = wandb.Artifact(
                name="validated-data",
                type="dataset",
                description="Dataset after schema validation and cleaning."
            )
            artifact.add_file(validated_data_path)
            wandb.log_artifact(artifact)
            logger.info(f"Logged validated data artifact to MLflow and W&B: {validated_data_path}")

        logger.info("--- Data Validation Step Completed Successfully ---")

    except Exception as e:
        logger.exception("Data validation step failed")
        if wandb_run:
            wandb.log({"status": "failed", "error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data validation pipeline step.")
    # The input artifact path is now read from the main config file
    config = load_config("conf/config.yaml")
    default_input = config.get("data_source", {}).get("raw_path", "data/raw/raw_data.csv")
    
    parser.add_argument(
        "--input-artifact", 
        default=default_input,
        help="Path to the raw data CSV file to be validated."
    )
    args = parser.parse_args()

    run_data_validation(args.input_artifact) 