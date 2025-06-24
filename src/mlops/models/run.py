import argparse
import logging
import os
import sys

import pandas as pd

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import mlflow

import wandb
from src.mlops.data_validation.data_validation import load_config
from src.mlops.models.models import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_model_training(input_artifact_dir: str):
    """
    Executes the model training step.
    - Loads processed training data.
    - Trains regression and classification models.
    - Logs model hyperparameters, metrics, and the trained models as artifacts.
    """
    logger.info("--- Starting Standalone Model Training Step ---")

    config = load_config("conf/config.yaml")

    # Set MLflow experiment
    mlflow_config = config.get("mlflow_tracking", {})
    experiment_name = mlflow_config.get(
        "experiment_name", "MLOps-Group-Project-Experiment"
    )
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to '{experiment_name}'")

    # Initialize a new W&B run
    wandb_config = config.get("wandb", {})
    wandb_run = wandb.init(
        project=wandb_config.get("project", "mlops-project"),
        entity=wandb_config.get("entity"),
        name="model_training-standalone",
        job_type="model-training",
    )

    try:
        with mlflow.start_run(run_name="model_training") as mlrun:
            # --- 1. Load Data ---
            logger.info(f"Loading processed data from: {input_artifact_dir}")

            X_train_reg = pd.read_csv(
                os.path.join(input_artifact_dir, "X_train_reg.csv")
            )
            y_train_reg = pd.read_csv(
                os.path.join(input_artifact_dir, "y_train_reg.csv")
            ).squeeze()

            X_train_class = pd.read_csv(
                os.path.join(input_artifact_dir, "X_train_class.csv")
            )
            y_train_class = pd.read_csv(
                os.path.join(input_artifact_dir, "y_train_class.csv")
            ).squeeze()

            mlflow.log_param("input_artifact_dir", input_artifact_dir)
            wandb.config.update({"input_artifact_dir": input_artifact_dir})

            # --- 2. Train Models ---
            model_trainer = ModelTrainer()

            # Train Regression Model
            logger.info("Training Linear Regression model...")
            reg_model = model_trainer.train_linear_regression(X_train_reg, y_train_reg)
            reg_predictions = reg_model.predict(X_train_reg)
            from sklearn.metrics import mean_squared_error

            reg_rmse = mean_squared_error(y_train_reg, reg_predictions) ** 0.5
            logger.info(f"Regression training RMSE: {reg_rmse}")

            # Train Classification Model
            logger.info("Training Logistic Regression model...")
            class_model = model_trainer.train_logistic_regression(
                X_train_class, y_train_class
            )
            class_predictions = class_model.predict(X_train_class)
            from sklearn.metrics import roc_auc_score

            try:
                class_roc_auc = roc_auc_score(y_train_class, class_predictions)
            except Exception:
                class_roc_auc = None
            logger.info(f"Classification training ROC AUC: {class_roc_auc}")

            # --- 3. Log Hyperparameters, Metrics, and Artifacts ---
            logger.info("Logging hyperparameters, metrics, and model artifacts...")

            # Log regression model details
            reg_params = (
                config.get("model", {}).get("linear_regression", {}).get("params", {})
            )
            mlflow.log_params({f"reg_{k}": v for k, v in reg_params.items()})
            mlflow.log_metric("train_reg_rmse", reg_rmse)

            reg_model_path = (
                config.get("model", {}).get("linear_regression", {}).get("save_path")
            )
            mlflow.log_artifact(reg_model_path, "regression-model")

            # Log classification model details
            class_params = (
                config.get("model", {}).get("logistic_regression", {}).get("params", {})
            )
            mlflow.log_params({f"class_{k}": v for k, v in class_params.items()})
            if class_roc_auc is not None:
                mlflow.log_metric("train_class_roc_auc", class_roc_auc)

            class_model_path = (
                config.get("model", {}).get("logistic_regression", {}).get("save_path")
            )
            mlflow.log_artifact(class_model_path, "classification-model")

            # Log to W&B
            wandb.config.update(
                {"regression_params": reg_params, "classification_params": class_params}
            )
            wandb.log(
                {
                    "regression_train_rmse": reg_rmse,
                    "classification_train_roc_auc": class_roc_auc,
                }
            )

            # Log training sample to W&B
            train_sample = X_train_reg.head(50).copy()
            train_sample["target_reg"] = y_train_reg.head(50)
            train_sample["target_class"] = y_train_class.head(50)
            wandb.log({"train_sample_rows": wandb.Table(dataframe=train_sample)})

            model_artifact = wandb.Artifact(
                "trained-models",
                type="model",
                description="Trained regression and classification models.",
            )
            model_artifact.add_file(reg_model_path)
            model_artifact.add_file(class_model_path)
            wandb.log_artifact(model_artifact)

            logger.info("All training details logged successfully.")

        logger.info("--- Model Training Step Completed Successfully ---")

    except Exception as e:
        logger.exception("Model training step failed")
        if wandb_run:
            wandb.log({"status": "failed", "error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the model training pipeline step."
    )
    config = load_config("conf/config.yaml")
    default_input = config.get("artifacts", {}).get(
        "processed_data_path", "data/processed/training_data"
    )

    parser.add_argument(
        "--input-artifact-dir",
        default=default_input,
        help="Path to the directory containing processed training data.",
    )
    args = parser.parse_args()

    run_model_training(args.input_artifact_dir)
