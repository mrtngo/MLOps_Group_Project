# main.py - New MLflow orchestrator for crypto prediction
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
from datetime import datetime
import wandb
import logging
import sys
from pathlib import Path

# Set up project paths
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv()

# Pipeline steps for crypto prediction
PIPELINE_STEPS = [
    "data_load",
    "data_validation", 
    "feature_engineering",
    "model",
    "evaluation",
    "inference",
]

# Steps that accept Hydra overrides
STEPS_WITH_OVERRIDES = {"model", "feature_engineering"}


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
        filemode="a"
    )

    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(log_level)
    console.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.getLogger("").addHandler(console)


@hydra.main(config_name="config", config_path="conf", version_base=None)
def main(cfg: DictConfig):
    """Main orchestrator for crypto prediction pipeline with Hydra configuration."""
    # Setup logging first
    setup_logging(cfg)
    logger = logging.getLogger("CryptoMLOps")
    
    logger.info("🚀 Starting Crypto MLOps Pipeline with Hydra configuration")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Set W&B environment variables
    os.environ["WANDB_PROJECT"] = cfg.main.WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = cfg.main.WANDB_ENTITY

    run_name = f"crypto_orchestrator_{datetime.now():%Y%m%d_%H%M%S}"
    
    try:
        # Initialize W&B run for orchestrator
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="orchestrator",
            name=run_name,
            config=dict(cfg),
            tags=["crypto", "orchestrator"]
        )
        logger.info(f"Started WandB run: {run.name}")

        # Parse which steps to run
        steps_raw = cfg.main.steps
        active_steps = [s.strip() for s in steps_raw.split(",") if s.strip()] \
            if steps_raw != "all" else PIPELINE_STEPS

        # Get hydra overrides for applicable steps
        hydra_override = cfg.main.hydra_options if hasattr(
            cfg.main, "hydra_options") else ""

        logger.info(f"Running crypto pipeline steps: {active_steps}")
        
        # Create output directories
        for dir_path in ["data/processed", "models", "logs", "plots"]:
            os.makedirs(dir_path, exist_ok=True)

        # For now, run the existing pipeline (you can update this later)
        logger.info("🔄 Running your existing pipeline...")
        
        # Import your existing main function
        from mlops.main import run_full_pipeline
        
        # Convert dates if provided
        start_date = cfg.data_source.get("start_date", "2023-01-01")
        end_date = cfg.data_source.get("end_date", "2023-12-31")
        
        # Run your existing pipeline
        run_full_pipeline(start_date, end_date)

        logger.info("🎉 Crypto MLOps pipeline completed successfully!")
        
        # Log pipeline summary to W&B
        wandb.summary.update({
            "pipeline_status": "success",
            "steps_completed": len(active_steps),
            "steps_list": active_steps,
        })

    except Exception as e:
        logger.error(f"❌ Crypto pipeline failed: {e}")
        if 'run' in locals():
            wandb.summary.update({
                "pipeline_status": "failed",
                "error_message": str(e)
            })
        raise
    finally:
        # Cleanup W&B run
        if 'run' in locals():
            wandb.finish()
            logger.info("Finished W&B orchestrator run")


# CLI interface for backward compatibility
def cli_main():
    """CLI interface for backward compatibility with original main.py."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto MLOps Pipeline")
    parser.add_argument("--stage", choices=["all", "training", "evaluation", "inference"], 
                       default="all", help="Pipeline stage")
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
        overrides.append("main.steps=data_load,data_validation,feature_engineering,model")
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