import argparse
import logging
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

import mlflow
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from src.mlops.data_validation.data_validation import load_config
from src.mlops.features.features import (
    define_features_and_label,
    create_price_direction_label
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_feature_engineering(input_artifact: str):
    """
    Executes the feature engineering step.
    - Loads validated data.
    - Creates new features (e.g., price direction label).
    - Logs feature distributions.
    - Logs the feature-engineered data as a new artifact.
    """
    logger.info("--- Starting Standalone Feature Engineering Step ---")

    # Find the project root directory (where conf/config.yaml is located)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..')
    config_path = os.path.join(project_root, 'conf', 'config.yaml')
    
    config = load_config(config_path)

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
        name="feature_engineering-standalone",
        job_type="feature-engineering"
    )

    try:
        with mlflow.start_run(run_name="feature_engineering") as mlrun:
            # --- 1. Load Data ---
            logger.info(f"Loading validated data from: {input_artifact}")
            df = pd.read_csv(input_artifact)
            # MLflow already logs the input artifact parameter, no need to do it manually
            # mlflow.log_param("input_artifact", input_artifact)
            # wandb.config.update({"input_artifact": input_artifact})

            # --- 2. Feature Engineering ---
            feature_cols, label_col = define_features_and_label(config)
            df_with_features = create_price_direction_label(df, label_col)
            logger.info("Feature engineering complete.")

            # --- 3. Log Visualizations ---
            logger.info("Generating and logging feature distribution plots to W&B...")
            num_features = len(feature_cols)
            num_cols = 4
            num_rows = (num_features + num_cols - 1) // num_cols
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 4))
            axes = axes.flatten()

            for i, feature in enumerate(feature_cols):
                if feature in df_with_features.columns:
                    sns.histplot(df_with_features[feature], kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {feature}')
            
            for i in range(num_features, len(axes)):
                fig.delaxes(axes[i])

            plt.tight_layout()
            wandb.log({"initial_feature_distributions": wandb.Image(plt)})
            plt.close(fig)
            logger.info("Visualizations logged.")

            # --- 4. Log Output Artifact ---
            output_path = config.get("artifacts", {}).get("feature_engineered_path", "data/processed/feature_engineered_data.csv")
            # Use absolute path for saving data
            output_path = os.path.join(project_root, output_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_with_features.to_csv(output_path, index=False)
            
            mlflow.log_artifact(output_path, "feature-engineered-data")
            
            artifact = wandb.Artifact(
                name="feature-engineered-data",
                type="dataset",
                description="Dataset after adding engineered features."
            )
            artifact.add_file(output_path)
            wandb.log_artifact(artifact)
            logger.info(f"Logged feature-engineered data artifact to: {output_path}")

        logger.info("--- Feature Engineering Step Completed Successfully ---")

    except Exception as e:
        logger.exception("Feature engineering step failed")
        if wandb_run:
            wandb.log({"status": "failed", "error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the feature engineering pipeline step.")
    # Find the project root directory and load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..', '..', '..')
    config_path = os.path.join(project_root, 'conf', 'config.yaml')
    config = load_config(config_path)
    
    default_input = config.get("data_source", {}).get("processed_path", "data/processed/validated_data.csv")
    # Use absolute path for default input
    default_input = os.path.join(project_root, default_input)
    
    parser.add_argument(
        "--input-artifact", 
        default=default_input,
        help="Path to the validated data CSV file."
    )
    args = parser.parse_args()

    run_feature_engineering(args.input_artifact) 