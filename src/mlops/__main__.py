"""Entry point of the package."""

# %% IMPORTS

from typing import Dict
import logging
import os
import argparse
import sys
import pandas as pd
from mlops import scripts
# from evaluation.evaluation import evaluate_classification, generate_report

import yaml

# from data_load.data_load import

from data_validation.data_validation import validate_data

# from models.models import

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


# %% MAIN
if __name__ == "__main__":
    scripts.main()
