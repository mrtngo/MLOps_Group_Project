import argparse
import logging
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import mlflow

import wandb
from src.mlops.data_load.data_load import load_config
from src.mlops.evaluation.evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def run_evaluation(model_artifact_path: str, test_data_dir: str):
    """
    Executes the model evaluation step.
    - Loads a trained model and test data.
    - Evaluates the model's performance.
    - Logs metrics and visualizations (e.g., confusion matrix, ROC curve).
    """
    logger.info("--- Starting Standalone Model Evaluation Step ---")

    # Find the project root directory and load config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..", "..")
    config_path = os.path.join(project_root, "conf", "config.yaml")
    config = load_config(config_path)

    # Set MLflow and W&B experiment
    mlflow_config = config.get("mlflow_tracking", {})
    experiment_name = mlflow_config.get(
        "experiment_name", "MLOps-Group-Project-Experiment"
    )
    mlflow.set_experiment(experiment_name)

    wandb_config = config.get("wandb", {})
    wandb.init(
        project=wandb_config.get("project", "mlops-project"),
        name="model_evaluation-standalone",
        config=config,
    )

    with mlflow.start_run(run_name="model_evaluation") as mlrun:
        logger.info(f"Evaluating model from: {model_artifact_path}")
        logger.info(f"Using test data from: {test_data_dir}")

        evaluator = ModelEvaluator(
            model_path=model_artifact_path, test_data_dir=test_data_dir, config=config
        )

        # The evaluator now handles both reg and class models if the data is present
        # We need to know which model we're evaluating based on the artifact path
        if "linear_regression" in model_artifact_path:
            reg_metrics = evaluator.evaluate_regression()
            if reg_metrics:
                logger.info(f"Regression Evaluation Metrics: {reg_metrics}")
                mlflow.log_metrics(
                    {f"regression_eval_{k}": v for k, v in reg_metrics.items()}
                )
                wandb.log({f"regression_eval_{k}": v for k, v in reg_metrics.items()})

        elif "logistic_regression" in model_artifact_path:
            class_metrics, plots, sample_df = evaluator.evaluate_classification()
            if class_metrics:
                logger.info(f"Classification Evaluation Metrics: {class_metrics}")
                mlflow.log_metrics(
                    {f"classification_eval_{k}": v for k, v in class_metrics.items()}
                )
                wandb.log(
                    {f"classification_eval_{k}": v for k, v in class_metrics.items()}
                )

            # Log plots to MLflow and W&B
            for plot_name, plot_path in plots.items():
                mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")
                wandb.log({plot_name: wandb.Image(plot_path)})
                logger.info(f"Logged {plot_name} to MLflow and W&B.")

            # Log sample predictions table to W&B
            if not sample_df.empty:
                wandb.log({"sample_predictions": wandb.Table(dataframe=sample_df)})
                logger.info("Logged sample predictions to W&B.")

        logger.info("--- Model Evaluation Step Completed Successfully ---")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model evaluation script.")
    parser.add_argument(
        "--model-artifact", type=str, required=True, help="Path to the model artifact."
    )
    parser.add_argument(
        "--test-data-path",
        type=str,
        required=True,
        help="Path to the test data directory.",
    )
    args = parser.parse_args()
    run_evaluation(args.model_artifact, args.test_data_path)
