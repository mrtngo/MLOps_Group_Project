import subprocess
import sys
import os
import pytest
from unittest import mock


def test_models_run_default(monkeypatch, tmp_path):
    script = os.path.join("src", "mlops", "models", "run.py")
    # Patch load_config to avoid file dependency
    with mock.patch(
        "src.mlops.data_validation.data_validation.load_config",
        return_value={"artifacts": {"processed_data_path": str(tmp_path)}},
    ):
        os.makedirs(tmp_path, exist_ok=True)
        # Write minimal valid CSVs for model training
        feature_cols = [
            "ETHUSDT_price",
            "BNBUSDT_price",
            "XRPUSDT_price",
            "ADAUSDT_price",
            "SOLUSDT_price",
            "BTCUSDT_funding_rate",
            "ETHUSDT_funding_rate",
            "BNBUSDT_funding_rate",
            "XRPUSDT_funding_rate",
            "ADAUSDT_funding_rate",
            "SOLUSDT_funding_rate",
        ]
        # X_train_reg.csv and X_train_class.csv
        for fname in ["X_train_reg.csv", "X_train_class.csv"]:
            with open(os.path.join(tmp_path, fname), "w") as f:
                f.write(",".join(feature_cols) + "\n")
                f.write(",".join(["1"] * len(feature_cols)) + "\n")
                f.write(",".join(["2"] * len(feature_cols)) + "\n")
        # y_train_reg.csv and y_train_class.csv
        for fname in ["y_train_reg.csv", "y_train_class.csv"]:
            with open(os.path.join(tmp_path, fname), "w") as f:
                f.write("y\n1\n2\n")
        result = subprocess.run(
            [sys.executable, script, "--input-artifact-dir", str(tmp_path)],
            capture_output=True,
        )
        assert result.returncode == 0


def test_models_run_missing_input():
    script = os.path.join("src", "mlops", "models", "run.py")
    result = subprocess.run(
        [sys.executable, script, "--input-artifact-dir", "not_a_real_dir"],
        capture_output=True,
    )
    assert result.returncode != 0
