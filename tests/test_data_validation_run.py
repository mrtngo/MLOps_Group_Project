import os
import sys
import json
import pandas as pd
import pytest
import yaml

# Functions under test
from src.mlops.data_validation.run import _html_from_report, run_data_validation

class DummyMLflowRun:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): pass

@pytest.fixture(autouse=True)
def patch_dependencies(tmp_path, monkeypatch):
    """
    Monkey-patch external dependencies:
      - load_config
      - mlflow.set_experiment, mlflow.start_run, mlflow.log_artifact
      - wandb.init, wandb.Html, wandb.Table, wandb.log, wandb.Artifact, wandb.finish
      - pandas.read_csv
      - os.path.exists
      - chdir to tmp project root
    """
    # Build a fake project root structure
    project_root = tmp_path / "project"
    (project_root / "conf").mkdir(parents=True)
    config_path = project_root / "conf" / "config.yaml"
    config = {
        "mlflow_tracking": {"experiment_name": "test_exp"},
        "wandb": {"project": "test_proj"},
        "data_validation": {
            "schema": {"columns": [{"name": "col1", "dtype": "int64"}]},
            "missing_values_strategy": "drop",
            "on_error": "warn"
        },
        "data_source": {"processed_path": "data/validated.csv"}
    }
    config_path.write_text(yaml.safe_dump(config))
    # Patch load_config
    import src.mlops.data_load.data_load as dl
    monkeypatch.setattr(dl, "load_config", lambda path: config)
    # Patch mlflow
    import mlflow
    monkeypatch.setattr(mlflow, "set_experiment", lambda name: None)
    monkeypatch.setattr(mlflow, "start_run", lambda run_name=None: DummyMLflowRun())
    monkeypatch.setattr(mlflow, "log_artifact", lambda *args, **kwargs: None)
    # Patch wandb
    import wandb
    monkeypatch.setattr(wandb, "init", lambda **kwargs: None)
    monkeypatch.setattr(wandb, "Html", lambda html: html)
    monkeypatch.setattr(wandb.Table, "__init__", lambda self, dataframe: None)
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)
    # Correct mock for wandb.Artifact
    class MockArtifact:
        def __init__(self, name, type, description):
            self.name = name
            self.type = type
            self.description = description
        def add_file(self, path):
            pass
    monkeypatch.setattr(wandb, "Artifact", MockArtifact)
    monkeypatch.setattr(wandb, "log_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb, "finish", lambda: None)
    # Patch pandas.read_csv to return a sample DataFrame
    sample_df = pd.DataFrame({"col1": [1, 2, 3]})
    monkeypatch.setattr(pd, "read_csv", lambda path: sample_df)
    # Default os.path.exists to True unless specifically overridden
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    # Change cwd to project root
    monkeypatch.chdir(project_root)
    return config, project_root

@ pytest.mark.parametrize("report,expected_text", [  # noqa: E203
    ({}, "<p><b>Result:</b> unknown</p>"),
    ({"status": "pass", "issues": {"errors": [], "warnings": []}}, "<p><b>Result:</b> pass</p>"),
])
def test_html_from_report_basic(report, expected_text):
    html = _html_from_report(report)
    assert expected_text in html


def test_html_from_report_errors_warnings_and_missing():
    report = {
        "status": "fail",
        "issues": {"errors": ["E1"], "warnings": ["W1"]},
        "missing_values_summary": {"strategy": "impute", "missing_before": 2, "total_imputed": 2},
        "column_details": {"col1": {"status": "pass", "expected_type": "int64", "sample_values": ["1"]}}
    }
    html = _html_from_report(report)
    # Check errors and warnings
    assert "Errors: 1 | Warnings: 1" in html
    assert "<li>E1</li>" in html
    assert "<li>W1</li>" in html
    # Check missing values summary
    assert "<b>Strategy:</b> impute" in html
    assert "Total values imputed" in html
    # Check column details table header and value
    assert "Column Details" in html
    assert "col1" in html


def test_run_data_validation_missing_artifact(monkeypatch, caplog):
    # Setup exists to return False
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    with pytest.raises(SystemExit) as exc:
        run_data_validation("nonexistent.csv")
    assert exc.value.code == 1
    assert "Input artifact not found" in caplog.text
