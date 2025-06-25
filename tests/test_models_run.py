import subprocess
import sys
import os
import pytest
from unittest import mock


def test_models_run_missing_input():
    script = os.path.join("src", "mlops", "models", "run.py")
    result = subprocess.run(
        [sys.executable, script, "--input-artifact-dir", "not_a_real_dir"],
        capture_output=True,
    )
    assert result.returncode != 0
