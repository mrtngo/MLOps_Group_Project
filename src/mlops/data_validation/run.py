import argparse
import json
import logging
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import mlflow
import pandas as pd

import wandb
from src.mlops.data_load.data_load import load_config
from src.mlops.data_validation.data_validation import validate_data

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _html_from_report(report: dict) -> str:
    """Create a simple HTML summary table from the validation report."""
    lines = ["<h2>Data Validation Report</h2>"]
    result = report.get("status", "unknown")
    lines.append(f"<p><b>Result:</b> {result}</p>")

    issues = report.get("issues", {})
    errors = issues.get("errors", [])
    warnings = issues.get("warnings", [])

    lines.append(f"<p>Errors: {len(errors)} | Warnings: {len(warnings)}</p>")

    if errors:
        lines.append("<h3>Errors</h3><ul>")
        for e in errors:
            lines.append(f"<li>{e}</li>")
        lines.append("</ul>")

    if warnings:
        lines.append("<h3>Warnings</h3><ul>")
        for w in warnings:
            lines.append(f"<li>{w}</li>")
        lines.append("</ul>")

    # Add missing values summary
    missing_summary = report.get("missing_values_summary")
    if missing_summary:
        lines.append("<h3>Missing Values Summary</h3>")
        strategy = missing_summary.get("strategy")
        lines.append(f"<p><b>Strategy:</b> {strategy}</p>")
        lines.append("<ul>")
        lines.append(
            f"<li>Missing values before handling: {missing_summary.get('missing_before', 0)}</li>"
        )
        if strategy == "impute":
            lines.append(
                f"<li>Total values imputed: {missing_summary.get('total_imputed', 0)}</li>"
            )
        elif strategy == "drop":
            lines.append(
                f"<li>Rows dropped: {missing_summary.get('rows_dropped', 0)}</li>"
            )
        lines.append("</ul>")

    details = report.get("column_details", {})
    if details:
        lines.append("<h3>Column Details</h3>")
        keys = set()
        for d in details.values():
            keys.update(d.keys())
        columns = ["column", "status", "expected_type", "sample_values"] + sorted(
            list(keys - {"status", "expected_type", "sample_values"})
        )

        lines.append(
            "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        )
        lines.append("<tr style='background-color: #f2f2f2;'>")
        for c in columns:
            lines.append(f"<th style='padding: 8px; text-align: left;'>{c}</th>")
        lines.append("</tr>")

        for i, (col, d) in enumerate(details.items()):
            row_style = (
                "background-color: #ffffff;"
                if i % 2 == 0
                else "background-color: #f9f9f9;"
            )
            lines.append(f"<tr style='{row_style}'>")
            for c in columns:
                val = col if c == "column" else d.get(c, "")
                if isinstance(val, list):
                    val = ", ".join(map(str, val))  # Join list items for display
                lines.append(
                    f"<td style='padding: 8px; border: 1px solid #ddd;'>{val}</td>"
                )
            lines.append("</tr>")
        lines.append("</table>")

    return "\n".join(lines)


def run_data_validation(input_artifact: str):
    """
    Executes the data validation step.
    - Loads data from an input artifact.
    - Validates it against a schema.
    - Logs summary stats and a rich HTML report to W&B.
    - Logs the validated data as a new artifact.
    """
    logger.info("--- Starting Standalone Data Validation Step ---")

    # Find the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..", "..")
    config_path = os.path.join(project_root, "conf", "config.yaml")

    config = load_config(config_path)

    # Set MLflow experiment
    mlflow_config = config.get("mlflow_tracking", {})
    experiment_name = mlflow_config.get(
        "experiment_name", "MLOps-Group-Project-Experiment"
    )
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment set to '{experiment_name}'")

    # Initialize W&B
    wandb_config = config.get("wandb", {})
    wandb.init(
        project=wandb_config.get("project", "mlops-project"),
        name="data_validation-standalone",
        config=config,
        job_type="data_validation",
    )

    with mlflow.start_run(run_name="data_validation") as mlrun:
        # --- 1. Load Data ---
        logger.info(f"Loading raw data from: {input_artifact}")
        if not os.path.exists(input_artifact):
            logger.error(
                f"Input artifact not found at {input_artifact}. Please run the data_load step first."
            )
            sys.exit(1)
        df = pd.read_csv(input_artifact)

        # --- 2. Validate Data ---
        logger.info("Validating data against the schema...")
        validation_config = config.get("data_validation", {})
        df_validated, report = validate_data(df, validation_config)
        logger.info(
            f"Data validation completed. Shape after validation: {df_validated.shape}"
        )

        # --- 3. Log Validation Report ---
        report_path = os.path.join(project_root, "reports", "validation_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        mlflow.log_artifact(report_path, "validation-report")

        # Generate and log HTML report to W&B
        html_report = _html_from_report(report)
        wandb.log({"validation_report": wandb.Html(html_report)})
        logger.info("Logged HTML validation report to W&B.")

        # --- 4. Log Summary Stats and Sample Data ---
        logger.info(
            "Generating and logging summary statistics and sample rows to W&B..."
        )
        summary_stats = df_validated.describe().to_dict()
        sample_rows = df_validated.head(20)
        wandb.log(
            {
                "validated_data_summary": summary_stats,
                "validated_data_sample": wandb.Table(dataframe=sample_rows),
            }
        )
        logger.info("Successfully logged W&B Tables.")

        # --- 5. Log Validated Data Artifact ---
        output_path = os.path.join(
            project_root, config.get("data_source", {}).get("processed_path")
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_validated.to_csv(output_path, index=False)

        mlflow.log_artifact(output_path, artifact_path="validated_data")

        artifact = wandb.Artifact(
            name="validated_data",
            type="dataset",
            description="Data that has been cleaned and validated against the schema.",
        )
        artifact.add_file(output_path)
        wandb.log_artifact(artifact)

        logger.info(f"Logged validated data artifact to MLflow and W&B: {output_path}")

        logger.info("--- Data Validation Step Completed Successfully ---")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data validation script.")
    parser.add_argument(
        "--input-artifact",
        type=str,
        required=True,
        help="Path to the raw data CSV artifact.",
    )
    args = parser.parse_args()
    run_data_validation(input_artifact=args.input_artifact)
