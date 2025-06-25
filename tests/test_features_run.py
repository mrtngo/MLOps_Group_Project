import subprocess
import sys
import os
import pytest
from unittest import mock

def test_features_run_default(monkeypatch, tmp_path):
    script = os.path.join('src', 'mlops', 'features', 'run.py')
    # Patch load_config to avoid file dependency
    with mock.patch('src.mlops.data_validation.data_validation.load_config', return_value={"data_source": {"processed_path": str(tmp_path / "dummy.csv")}}):
        with open(tmp_path / "dummy.csv", "w") as f:
            f.write("timestamp,BTCUSDT_price\n2024-01-01,100\n")
        result = subprocess.run([sys.executable, script, "--input-artifact", str(tmp_path / "dummy.csv")], capture_output=True)
        assert result.returncode == 0

def test_features_run_missing_input():
    script = os.path.join('src', 'mlops', 'features', 'run.py')
    result = subprocess.run([sys.executable, script, "--input-artifact", "not_a_real_file.csv"], capture_output=True)
    assert result.returncode != 0 