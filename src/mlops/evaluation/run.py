import argparse
import logging
import os
import sys
import pickle
import pandas as pd
import mlflow
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, confusion_matrix, classification_report, f1_score, roc_curve

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from src.mlops.data_validation.data_validation import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_evaluation(input_artifact_dir: str):
    logger.info("--- Starting Standalone Model Evaluation Step ---")
    config = load_config("conf/config.yaml")

    mlflow_config = config.get("mlflow_tracking", {})
    experiment_name = mlflow_config.get("experiment_name", "MLOps-Group-Project-Experiment")
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to '{experiment_name}'")

    wandb_config = config.get("wandb", {})
    wandb_run = wandb.init(
        project=wandb_config.get("project", "mlops-project"),
        entity=wandb_config.get("entity"),
        name="evaluation-standalone",
        job_type="evaluation"
    )

    try:
        with mlflow.start_run(run_name="evaluation") as mlrun:
            # --- 1. Load Data and Models ---
            logger.info(f"Loading test data from: {input_artifact_dir}")
            X_test_reg = pd.read_csv(os.path.join(input_artifact_dir, "X_test_reg.csv"))
            y_test_reg = pd.read_csv(os.path.join(input_artifact_dir, "y_test_reg.csv")).squeeze()
            X_test_class = pd.read_csv(os.path.join(input_artifact_dir, "X_test_class.csv"))
            y_test_class = pd.read_csv(os.path.join(input_artifact_dir, "y_test_class.csv")).squeeze()

            reg_model_path = config.get("model", {}).get("linear_regression", {}).get("save_path")
            class_model_path = config.get("model", {}).get("logistic_regression", {}).get("save_path")
            with open(reg_model_path, 'rb') as f:
                reg_model = pickle.load(f)
            with open(class_model_path, 'rb') as f:
                class_model = pickle.load(f)

            # --- 2. Regression Evaluation ---
            reg_predictions = reg_model.predict(X_test_reg)
            reg_rmse = mean_squared_error(y_test_reg, reg_predictions) ** 0.5
            mlflow.log_metric("test_reg_rmse", reg_rmse)
            wandb.log({"test_reg_rmse": reg_rmse})

            # --- 3. Classification Evaluation ---
            class_predictions = class_model.predict(X_test_class)
            class_probs = class_model.predict_proba(X_test_class)[:, 1] if hasattr(class_model, "predict_proba") else None
            class_roc_auc = None
            if class_probs is not None:
                try:
                    class_roc_auc = roc_auc_score(y_test_class, class_probs)
                except Exception:
                    class_roc_auc = None
            class_acc = accuracy_score(y_test_class, class_predictions)
            class_f1 = f1_score(y_test_class, class_predictions)
            mlflow.log_metric("test_class_roc_auc", class_roc_auc if class_roc_auc is not None else -1)
            mlflow.log_metric("test_class_accuracy", class_acc)
            mlflow.log_metric("test_class_f1", class_f1)
            wandb.log({
                "test_class_roc_auc": class_roc_auc,
                "test_class_accuracy": class_acc,
                "test_class_f1": class_f1
            })

            # --- 4. Confusion Matrix ---
            cm = confusion_matrix(y_test_class, class_predictions)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            plt.tight_layout()
            wandb.log({"confusion_matrix": wandb.Image(fig)})
            plt.close(fig)

            # --- 5. ROC Curve ---
            if class_probs is not None:
                fpr, tpr, _ = roc_curve(y_test_class, class_probs)
                plt.figure(figsize=(6, 5))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {class_roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend(loc='lower right')
                wandb.log({"roc_curve": wandb.Image(plt)})
                plt.close()

            # --- 6. Log Test Sample Table ---
            test_sample = X_test_reg.head(50).copy()
            test_sample['target_reg'] = y_test_reg.head(50)
            test_sample['target_class'] = y_test_class.head(50)
            wandb.log({"test_sample_rows": wandb.Table(dataframe=test_sample)})

            # --- 7. Save and Log Metrics Artifact ---
            metrics = {
                "test_reg_rmse": reg_rmse,
                "test_class_roc_auc": class_roc_auc,
                "test_class_accuracy": class_acc,
                "test_class_f1": class_f1
            }
            metrics_path = os.path.join(input_artifact_dir, "test_metrics.json")
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            mlflow.log_artifact(metrics_path, "test-metrics")
            artifact = wandb.Artifact("test-metrics", type="metrics")
            artifact.add_file(metrics_path)
            wandb.log_artifact(artifact)

            logger.info("All evaluation metrics and visualizations logged successfully.")
        logger.info("--- Model Evaluation Step Completed Successfully ---")

    except Exception as e:
        logger.exception("Evaluation step failed")
        if wandb_run:
            wandb.log({"status": "failed", "error": str(e)})
        raise
    finally:
        if wandb_run:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the model evaluation pipeline step.")
    config = load_config("conf/config.yaml")
    default_input = config.get("artifacts", {}).get("processed_data_path", "data/processed/training_data")
    parser.add_argument(
        "--input-artifact-dir",
        default=default_input,
        help="Path to the directory containing processed test data."
    )
    args = parser.parse_args()
    run_evaluation(args.input_artifact_dir) 