# main.py - New MLflow orchestrator for crypto prediction
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import hydra
import mlflow
import pandas as pd
from dotenv import load_dotenv
from omegaconf import DictConfig

import wandb

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv()

# Define the sequence of steps in the pipeline
PIPELINE_STEPS = [
    "data_load",
    "data_validation",
    "features",
    "preprocess",
    "models",
    "evaluation",
    "inference",
]

# Define steps that can receive Hydra configuration overrides via MLflow
# This is useful for hyperparameter tuning during the model training step
STEPS_WITH_OVERRIDES = {"models"}


def setup_logging(config: DictConfig) -> None:
    """Setup logging configuration from Hydra config."""
    log_level = getattr(logging, config.logging.get("level", "INFO").upper())
    log_format = config.logging.get(
        "format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    date_format = config.logging.get("datefmt", "%Y-%m-%d %H:%M:%S")
    log_file = config.logging.get("log_file", "logs/main.log")

    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        filename=log_file,
        filemode="a",
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger("").addHandler(console)


@hydra.main(config_name="config", config_path="conf", version_base=None)
def main(cfg: DictConfig):
    """
    Main orchestrator for the MLOps pipeline.

    Initializes a W&B run and then executes each pipeline step
    as a separate MLflow Project.
    """
    # Set W&B environment variables from the Hydra config
    os.environ["WANDB_PROJECT"] = cfg.main.WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = cfg.main.WANDB_ENTITY

    # Initialize a parent W&B run for the orchestrator
    run_name = f"orchestrator_{datetime.now():%Y%m%d_%H%M%S}"
    run = wandb.init(
        project=cfg.main.WANDB_PROJECT,
        entity=cfg.main.WANDB_ENTITY,
        job_type="orchestrator",
        name=run_name,
    )
    print(f"✅ Started W&B orchestrator run: {run.name}")

    # Set the MLflow experiment for the orchestrator
    mlflow_config = cfg.get("mlflow_tracking", {})
    experiment_name = mlflow_config.get(
        "experiment_name", "MLOps-Group-Project-Experiment"
    )
    mlflow.set_experiment(experiment_name)
    print(f"✅ Set MLflow experiment to: '{experiment_name}'")

    # Determine which steps to run
    steps_raw = cfg.main.steps
    print(f"DEBUG: steps_raw = {steps_raw}, type = {type(steps_raw)}")

    if isinstance(steps_raw, str) and steps_raw != "all":
        active_steps = [s.strip() for s in steps_raw.split(",")]
    elif hasattr(steps_raw, "__iter__") and not isinstance(steps_raw, str):
        # Handle both regular lists and OmegaConf ListConfig
        active_steps = list(steps_raw)
    else:
        active_steps = PIPELINE_STEPS

    print(f"DEBUG: active_steps = {active_steps}")
    print(f"DEBUG: PIPELINE_STEPS = {PIPELINE_STEPS}")

    # Get any Hydra command-line overrides
    hydra_override = cfg.main.get("hydra_options", "")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Define artifact paths to pass between steps
        artifacts = {
            "raw_data": os.path.join(
                hydra.utils.get_original_cwd(), cfg.data_source.raw_path
            ),
            "validated_data": os.path.join(
                hydra.utils.get_original_cwd(), cfg.data_source.processed_path
            ),
            "feature_engineered_data": os.path.join(
                hydra.utils.get_original_cwd(),
                "data/processed/feature_engineered_data.csv",
            ),
            "processed_data_path": os.path.join(
                hydra.utils.get_original_cwd(), "data/processed/training_data"
            ),
            "model_path": os.path.join(
                hydra.utils.get_original_cwd(), "models/logistic_regression.pkl"
            ),
            "test_data_path_class": os.path.join(
                hydra.utils.get_original_cwd(),
                "data/processed/training_data/X_test_class.csv",
            ),
        }

        for step in active_steps:
            if step not in PIPELINE_STEPS:
                logging.warning(f"Skipping unknown step: {step}")
                continue

            print(f"▶️  Running step: '{step}'...")

            # Define parameters for the current step
            params = {}
            if step == "data_validation":
                params["input_artifact"] = artifacts["raw_data"]
            elif step == "features":
                params["input_artifact"] = artifacts["validated_data"]
            elif step == "preprocess":
                params["input_artifact"] = artifacts["feature_engineered_data"]
            elif step == "models":
                params["input_artifact"] = artifacts["processed_data_path"]
            elif step == "evaluation":
                params["model_artifact"] = artifacts["model_path"]
                params["test_data_path"] = artifacts["processed_data_path"]
            elif step == "inference":
                params["model_artifact"] = artifacts["model_path"]
                params["inference_data"] = artifacts["test_data_path_class"]

            # MLflow will now use the conda.yaml file specified in each
            # step's entry point in the root MLproject file.
            mlflow.run(".", entry_point=step, parameters=params)

            print(f"✅ Step '{step}' finished.")

    print("🎉 Pipeline execution complete.")
    wandb.finish()


# CLI interface for backward compatibility
def cli_main():
    """CLI interface for backward compatibility with original main.py."""
    import argparse

    parser = argparse.ArgumentParser(description="Crypto MLOps Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "training", "evaluation", "inference"],
        default="all",
        help="Pipeline stage",
    )
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-csv", help="Output file for inference")

    args = parser.parse_args()

    # Map CLI args to Hydra overrides
    overrides = []
    if args.start_date:
        overrides.append(f"data_source.start_date={args.start_date}")
    if args.end_date:
        overrides.append(f"data_source.end_date={args.end_date}")
    if args.stage == "training":
        overrides.append(
            "main.steps=data_load,data_validation,feature_engineering,model"
        )
    elif args.stage == "evaluation":
        overrides.append("main.steps=evaluation")
    elif args.stage == "inference":
        overrides.append("main.steps=inference")
        if args.output_csv:
            overrides.append(f"inference.output_csv={args.output_csv}")

    # Add overrides to sys.argv for Hydra
    sys.argv = ["main.py"] + overrides

    # Run main with Hydra
    main()


if __name__ == "__main__":
    # Check if we're running with Hydra overrides or CLI args
    if any(arg.startswith("--") for arg in sys.argv[1:]):
        cli_main()
    else:
        main()
