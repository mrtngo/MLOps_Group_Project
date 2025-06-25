import subprocess
import sys
import os
import pytest
from unittest import mock


def test_preproccess_run_missing_input():
    script = os.path.join("src", "mlops", "preproccess", "run.py")
    result = subprocess.run(
        [sys.executable, script, "--input-artifact", "not_a_real_file.csv"],
        capture_output=True,
    )
    assert result.returncode != 0
