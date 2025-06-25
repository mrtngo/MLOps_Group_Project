""" import subprocess
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
    assert result.returncode != 0 """

import os
import tempfile
import pandas as pd
import pytest
from unittest import mock
import sys
import shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.mlops.preproccess.run import run_preprocessing

# --- Fixtures ---

@pytest.fixture
def mock_config():
    return {
        "mlflow_tracking": {"experiment_name": "test-experiment"},
        "wandb": {"project": "test-project", "entity": "test-entity"},
        "target": "price",
        "preprocessing": {
            "sampling": {"method": "smote"},
        },
        "artifacts": {
            "processed_data_path": "test_output",
            "preprocessing_pipeline": "test_models/preprocessing_pipeline.pkl",
        },
    }

@pytest.fixture
def dummy_csv_file():
    # Create a larger dataset to avoid SMOTE issues
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feature2": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "price": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "price_direction": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    
    # Use a more reliable temporary file approach
    fd, temp_file_path = tempfile.mkstemp(suffix='.csv', text=True)
    
    try:
        # Close the file descriptor and write using pandas
        os.close(fd)
        df.to_csv(temp_file_path, index=False)
        
        # Verify the file was written correctly
        verification_df = pd.read_csv(temp_file_path)
        assert len(verification_df) == 10, "Data not written correctly"
        assert list(verification_df.columns) == ["feature1", "feature2", "price", "price_direction"]
        
        yield temp_file_path
        
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- Mocks for external systems ---

@pytest.fixture(autouse=True)
def mock_dependencies():
    # Create a mock for open that only affects specific files, not our test CSV
    original_open = open
    
    def selective_mock_open(filename, *args, **kwargs):
        # Only mock specific files, not our test CSV files
        if filename.endswith('.pkl') or 'preprocessing_pipeline' in filename:
            return mock.mock_open()(*args, **kwargs)
        else:
            return original_open(filename, *args, **kwargs)
    
    with mock.patch("src.mlops.preproccess.run.load_config") as mock_config_loader, \
         mock.patch("src.mlops.preproccess.run.mlflow"), \
         mock.patch("src.mlops.preproccess.run.wandb"), \
         mock.patch("src.mlops.preproccess.run.plt"), \
         mock.patch("src.mlops.preproccess.run.sns"), \
         mock.patch("os.makedirs"), \
         mock.patch("builtins.open", side_effect=selective_mock_open), \
         mock.patch("pickle.dump"):
        yield mock_config_loader

# --- Test Cases ---

@mock.patch("src.mlops.preproccess.run.define_features_and_label", return_value=(["feature1", "feature2"], "price"))
def test_run_preprocessing_success(mock_define, mock_dependencies, mock_config, dummy_csv_file):
    mock_dependencies.return_value = mock_config
    run_preprocessing(dummy_csv_file)

@mock.patch("src.mlops.preproccess.run.define_features_and_label", return_value=(["feature1", "feature2"], "price"))
def test_run_preprocessing_handles_smote_branch(mock_define, mock_dependencies, mock_config, dummy_csv_file):
    mock_config["preprocessing"]["sampling"]["method"] = "smote"
    mock_dependencies.return_value = mock_config
    run_preprocessing(dummy_csv_file)

@mock.patch("src.mlops.preproccess.run.define_features_and_label", return_value=(["feature1", "feature2"], "price"))
def test_run_preprocessing_handles_no_smote_branch(mock_define, mock_dependencies, mock_config, dummy_csv_file):
    mock_config["preprocessing"]["sampling"]["method"] = "none"
    mock_dependencies.return_value = mock_config
    run_preprocessing(dummy_csv_file)

def test_run_preprocessing_raises_on_invalid_file(mock_dependencies, mock_config):
    mock_dependencies.return_value = mock_config
    with pytest.raises(Exception):
        run_preprocessing("nonexistent_file.csv")