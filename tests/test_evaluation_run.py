import subprocess
import sys
import os
import pytest
from unittest import mock


def test_evaluation_run_missing_model():
    script = os.path.join("src", "mlops", "evaluation", "run.py")
    # Should fail due to missing model file
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--model-artifact",
            "not_a_real_model.pkl",
            "--test-data-path",
            "not_a_real_dir",
        ],
        capture_output=True,
    )
    assert result.returncode != 0
