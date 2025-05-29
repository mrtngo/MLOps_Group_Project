"""Entry point of the package."""

# %% IMPORTS

from typing import Dict
import logging
import os
import argparse
import sys
from src.mlops.data_load.data_load import fetch_data
from src.mlops.data_validation.data_validation import validate_data
from src.mlops.models.models import train_model
import pandas as pd
from mlops import scripts
# from evaluation.evaluation import evaluate_classification, generate_report

import yaml

# from inference.inferencer import run_inference

logger = logging.getLogger(__name__)


# helpers
def _setup_logging(cfg: Dict) -> None:
    """Configure root logger from config.yaml → logging section."""
    log_level = cfg.get("level", "INFO").upper()
    log_file = cfg.get("log_file", "logs/main.log")
    fmt = cfg.get(
        "format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    datefmt = cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=fmt,
        datefmt=datefmt,
        filename=log_file,
        filemode="a",
    )
    # echo to console as well
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt, datefmt))
    console.setLevel(getattr(logging, log_level, logging.INFO))
    logging.getLogger().addHandler(console)


def _load_config(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLOps pipeline orchestrator")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Optional .env with credentials / environment vars",
    )

    args = parser.parse_args()

    # 1 – config & logging -------------------------------------------------
    try:
        cfg = _load_config(args.config)
    except Exception as exc:
        print(f"[main] Unable to read config: {exc}", file=sys.stderr)
        sys.exit(1)

    _setup_logging(cfg.get("logging", {}))
    logger.info("Pipeline started | stage=%s", args.stage)

    try:
        # 2 – data loading + validation -----------------------------------
        if args.stage in ("all", "data"):
            df_raw = fetch_data()
            logger.info("Raw data loaded | shape=%s", df_raw.shape)
            df = validate_data(df_raw, cfg["data_validation"]["schema"], logger,
                               cfg["data_validation"]["missing_values_strategy"])
            train_model(df)
            generate_report(cfg)

        # 4 – batch inference --------------------------------------------
        if args.stage == "infer":
            if not args.input_csv or not args.output_csv:
                logger.error(
                    "Inference stage requires --input_csv and --output_csv")
                sys.exit(1)
            # Load config and input for validation
            input_df = None
            try:
                input_df = pd.read_csv(args.input_csv)
            except Exception as exc:
                logger.error(f"Could not load input CSV: {exc}")
                sys.exit(1)
            # Validate inference input
            validate_data(input_df, cfg)
            # Now run inference
            run_inference(args.input_csv, args.config, args.output_csv)

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)

    logger.info("Pipeline completed successfully")


# CLI wrapper
if __name__ == "__main__":
    main()
