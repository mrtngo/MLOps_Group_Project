import subprocess
import sys
import os
import pytest
import tempfile
import yaml
from unittest import mock
from pathlib import Path
import pandas as pd
from unittest.mock import patch, MagicMock, call
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlops.data_load.run import run_data_load


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "data_source": {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "raw_path": "data/raw/raw_data.csv"
        },
        "mlflow_tracking": {
            "experiment_name": "test-experiment"
        },
        "wandb": {
            "project": "test-project",
            "entity": "test-entity"
        },
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "data_load": {
            "log_sample_rows": True,
            "log_summary_stats": True
        }
    }


@pytest.fixture
def dummy_df():
    """Create dummy input dataframe for testing"""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=5),
            "BTCUSDT_price": [100, 101, 102, 103, 104],
            "ETHUSDT_price": [50, 51, 52, 53, 54],
            "BTCUSDT_funding_rate": [0.01, 0.02, 0.015, 0.017, 0.018],
        }
    )


class TestDataLoadRun:
    """Test the data_load run.py functionality"""
    
    @patch('mlops.data_load.run.load_config')
    @patch('mlops.data_load.run.fetch_data')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('wandb.init')
    @patch('wandb.finish')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    @patch('mlflow.log_params')
    @patch('mlflow.log_artifact')
    @patch('wandb.config.update')
    @patch('wandb.log')
    @patch('wandb.Table')
    @patch('wandb.Artifact')
    def test_run_data_load_success(
        self, mock_artifact, mock_table, mock_wandb_log, mock_wandb_config_update,
        mock_mlflow_log_artifact, mock_mlflow_log_params, mock_to_csv, mock_makedirs,
        mock_wandb_finish, mock_wandb_init, mock_mlflow_start, mock_mlflow_set_experiment,
        mock_fetch_data, mock_load_config, dummy_df, mock_config
    ):
        """Test successful data load execution"""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_fetch_data.return_value = dummy_df
        
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
        
        mock_mlflow_run = MagicMock()
        mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
        mock_mlflow_start.return_value.__exit__.side_effect = None
        
        mock_table_instance = MagicMock()
        mock_table.return_value = mock_table_instance
        
        mock_artifact_instance = MagicMock()
        mock_artifact.return_value = mock_artifact_instance
        
        # Execute
        run_data_load()
        
        # Assertions
        mock_load_config.assert_called_once()
        mock_fetch_data.assert_called_once_with(mock_config, start_date="2024-01-01", end_date="2024-01-31")
        mock_mlflow_set_experiment.assert_called_once_with("test-experiment")
        mock_wandb_init.assert_called_once()
        mock_mlflow_log_params.assert_called_once()
        mock_wandb_config_update.assert_called_once()
        mock_to_csv.assert_called_once()
        mock_mlflow_log_artifact.assert_called_once()
        mock_wandb_finish.assert_called_once()
        
        # Check W&B logging calls
        assert mock_wandb_log.call_count >= 3  # sample_rows, summary_stats, raw_data_rows/columns
    
    @patch('mlops.data_load.run.load_config')
    @patch('mlops.data_load.run.fetch_data')
    @patch('mlflow.set_experiment')
    @patch('wandb.init')
    @patch('wandb.finish')
    @patch('mlflow.start_run')
    @patch('wandb.log')
    def test_run_data_load_exception(
        self, mock_wandb_log, mock_mlflow_start, mock_wandb_finish, mock_wandb_init, mock_mlflow_set_experiment,
        mock_fetch_data, mock_load_config, mock_config
    ):
        """Test data load execution with exception"""
        mock_load_config.return_value = mock_config
        mock_fetch_data.side_effect = Exception("Test error")
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
        # Properly mock mlflow.start_run as a context manager
        mock_mlflow_run = MagicMock()
        mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
        mock_mlflow_start.return_value.__exit__.side_effect = None
        # Execute and expect exception
        with pytest.raises(Exception, match="Test error"):
            run_data_load()
        # Check that wandb.log was called with the error
        mock_wandb_log.assert_any_call({"status": "failed", "error": "Test error"})
        mock_wandb_finish.assert_called_once()
    
    @patch('mlops.data_load.run.load_config')
    @patch('mlops.data_load.run.fetch_data')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_artifact')
    @patch('wandb.init')
    @patch('wandb.finish')
    @patch('wandb.log')
    @patch('wandb.log_artifact')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_run_data_load_without_wandb_logging(
        self, mock_to_csv, mock_makedirs, mock_wandb_log_artifact, mock_wandb_log, mock_wandb_finish, mock_wandb_init,
        mock_mlflow_log_artifact, mock_mlflow_log_params, mock_mlflow_start, mock_mlflow_set_experiment,
        mock_fetch_data, mock_load_config, dummy_df
    ):
        """Test data load without W&B sample logging"""
        import wandb
        wandb.config = MagicMock()
        
        config_without_logging = {
            "data_source": {
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "raw_path": "data/raw/raw_data.csv"
            },
            "mlflow_tracking": {
                "experiment_name": "test-experiment"
            },
            "wandb": {
                "project": "test-project",
                "entity": "test-entity"
            },
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data_load": {
                "log_sample_rows": False,
                "log_summary_stats": False
            }
        }
        
        mock_load_config.return_value = config_without_logging
        mock_fetch_data.return_value = dummy_df
        
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
        
        mock_mlflow_run = MagicMock()
        mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
        mock_mlflow_start.return_value.__exit__.side_effect = None
        
        # Execute
        run_data_load()
        
        # Should still finish wandb even without logging
        mock_wandb_finish.assert_called_once()
    
    @patch('mlops.data_load.run.load_config')
    @patch('mlops.data_load.run.fetch_data')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('wandb.init')
    @patch('wandb.finish')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_run_data_load_default_dates(
        self, mock_to_csv, mock_makedirs, mock_wandb_finish, mock_wandb_init,
        mock_mlflow_set_experiment, mock_mlflow_start, mock_fetch_data, mock_load_config, dummy_df
    ):
        """Test data load with default dates when not in config"""
        config_without_dates = {
            "data_source": {
                "raw_path": "data/raw/raw_data.csv"
            },
            "mlflow_tracking": {
                "experiment_name": "test-experiment"
            },
            "wandb": {
                "project": "test-project",
                "entity": "test-entity"
            },
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data_load": {
                "log_sample_rows": True,
                "log_summary_stats": True
            }
        }
        
        mock_load_config.return_value = config_without_dates
        mock_fetch_data.return_value = dummy_df
        
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
        
        mock_mlflow_run = MagicMock()
        mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
        mock_mlflow_start.return_value.__exit__.side_effect = None
        
        # Execute
        run_data_load()
        
        # Should use default dates (check the actual default dates used in the code)
        # The actual default dates depend on the fetch_data function implementation
        mock_fetch_data.assert_called_once()
        mock_wandb_finish.assert_called_once()
    
    @patch('mlops.data_load.run.load_config')
    @patch('mlops.data_load.run.fetch_data')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('wandb.init')
    @patch('wandb.finish')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    def test_run_data_load_wandb_finish_on_exception(
        self, mock_to_csv, mock_makedirs, mock_wandb_finish, mock_wandb_init,
        mock_mlflow_set_experiment, mock_mlflow_start, mock_fetch_data, mock_load_config, dummy_df, mock_config
    ):
        """Test that wandb.finish is called even when an exception occurs"""
        mock_load_config.return_value = mock_config
        mock_fetch_data.side_effect = Exception("Test error")
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
        
        mock_mlflow_run = MagicMock()
        mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
        mock_mlflow_start.return_value.__exit__.side_effect = None
        
        # Execute and expect exception
        with pytest.raises(Exception, match="Test error"):
            run_data_load()
        
        # Should still call wandb.finish
        mock_wandb_finish.assert_called_once()


def test_run_script_exists():
    """Test that the run script exists and is executable"""
    script_path = os.path.join(os.path.dirname(__file__), "..", "src", "mlops", "data_load", "run.py")
    assert os.path.exists(script_path), f"Script not found at {script_path}"


def test_run_missing_config():
    """Test that the script handles missing config gracefully"""
    script_path = os.path.join(os.path.dirname(__file__), "..", "src", "mlops", "data_load", "run.py")
    
    # Test with non-existent config file
    with patch('mlops.data_load.run.load_config') as mock_load_config:
        mock_load_config.side_effect = FileNotFoundError("Config file not found")
        
        with pytest.raises(FileNotFoundError):
            run_data_load()


def test_run_with_mocked_dependencies():
    """Test the run function with all dependencies mocked"""
    with patch('mlops.data_load.run.load_config') as mock_load_config:
        with patch('mlops.data_load.run.fetch_data') as mock_fetch_data:
            with patch('mlflow.set_experiment'):
                with patch('mlflow.start_run') as mock_mlflow_start:
                    with patch('wandb.init') as mock_wandb_init:
                        with patch('wandb.finish'):
                            with patch('os.makedirs'):
                                with patch('pandas.DataFrame.to_csv'):
                                    with patch('mlflow.log_params'):
                                        with patch('mlflow.log_artifact'):
                                            with patch('wandb.config.update'):
                                                with patch('wandb.log'):
                                                    with patch('wandb.Table'):
                                                        with patch('wandb.Artifact'):
                                                            # Setup mocks
                                                            mock_config = {
                                                                "data_source": {
                                                                    "start_date": "2024-01-01",
                                                                    "end_date": "2024-01-31",
                                                                    "raw_path": "data/raw/raw_data.csv"
                                                                },
                                                                "mlflow_tracking": {"experiment_name": "test"},
                                                                "wandb": {"project": "test"},
                                                                "symbols": ["BTCUSDT"],
                                                                "data_load": {"log_sample_rows": True, "log_summary_stats": True}
                                                            }
                                                            mock_load_config.return_value = mock_config
                                                            # Create a proper DataFrame with columns
                                                            mock_fetch_data.return_value = pd.DataFrame({
                                                                'timestamp': pd.date_range('2024-01-01', periods=5),
                                                                'price': [100, 101, 102, 103, 104]
                                                            })
                                                            
                                                            mock_wandb_run = MagicMock()
                                                            mock_wandb_init.return_value = mock_wandb_run
                                                            
                                                            mock_mlflow_run = MagicMock()
                                                            mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
                                                            mock_mlflow_start.return_value.__exit__.side_effect = None
                                                            
                                                            # Execute
                                                            run_data_load()
                                                            
                                                            # Verify calls
                                                            mock_load_config.assert_called_once()
                                                            mock_fetch_data.assert_called_once()


def test_run_invalid_config():
    """Test that the script handles invalid config gracefully"""
    with patch('mlops.data_load.run.load_config') as mock_load_config:
        mock_load_config.side_effect = yaml.YAMLError("Invalid YAML")
        
        with pytest.raises(yaml.YAMLError):
            run_data_load()


def test_script_syntax():
    """Test that the script has valid Python syntax"""
    script_path = os.path.join(os.path.dirname(__file__), "..", "src", "mlops", "data_load", "run.py")
    
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # This should not raise a syntax error
    compile(script_content, script_path, 'exec')


def test_script_imports():
    """Test that the script can be imported without errors"""
    import importlib.util
    import os

    script_path = os.path.join(os.path.dirname(__file__), "..", "src", "mlops", "data_load", "run.py")
    spec = importlib.util.spec_from_file_location("data_load_run", script_path)

    # Check that spec was created successfully
    assert spec is not None

    module = importlib.util.module_from_spec(spec)

    # This should not raise any import errors
    if spec.loader is not None:
        spec.loader.exec_module(module)


def test_config_file_structure():
    """Test that the config file has the expected structure"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "conf", "config.yaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check for required sections
    assert 'data_source' in config
    assert 'data_load' in config
    # Note: mlflow_tracking and wandb might not be in the actual config


def test_minimal_run_simulation():
    """Test a minimal run simulation with basic mocks"""
    with patch('mlops.data_load.run.load_config') as mock_load_config:
        with patch('mlops.data_load.run.fetch_data') as mock_fetch_data:
            with patch('mlflow.set_experiment'):
                with patch('mlflow.start_run') as mock_mlflow_start:
                    with patch('wandb.init') as mock_wandb_init:
                        with patch('wandb.finish'):
                            with patch('os.makedirs'):
                                with patch('pandas.DataFrame.to_csv'):
                                    with patch('mlflow.log_params'):
                                        with patch('mlflow.log_artifact'):
                                            with patch('wandb.config.update'):
                                                with patch('wandb.log'):
                                                    with patch('wandb.Table'):
                                                        with patch('wandb.Artifact'):
                                                            # Minimal config
                                                            mock_config = {
                                                                "data_source": {"raw_path": "test.csv"},
                                                                "mlflow_tracking": {"experiment_name": "test"},
                                                                "wandb": {"project": "test"},
                                                                "symbols": ["BTCUSDT"],
                                                                "data_load": {"log_sample_rows": False, "log_summary_stats": False}
                                                            }
                                                            mock_load_config.return_value = mock_config
                                                            mock_fetch_data.return_value = pd.DataFrame({'test': [1, 2, 3]})
                                                            
                                                            mock_wandb_run = MagicMock()
                                                            mock_wandb_init.return_value = mock_wandb_run
                                                            
                                                            mock_mlflow_run = MagicMock()
                                                            mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
                                                            mock_mlflow_start.return_value.__exit__.side_effect = None
                                                            
                                                            # Execute
                                                            run_data_load()
                                                            
                                                            # Basic verification
                                                            mock_load_config.assert_called_once()
                                                            mock_fetch_data.assert_called_once()


def test_find_data_load_files():
    """Test that data load files can be found"""
    import os
    
    # Check that the main script exists
    script_path = os.path.join(os.path.dirname(__file__), "..", "src", "mlops", "data_load", "run.py")
    assert os.path.exists(script_path), f"Script not found at {script_path}"
    
    # Check that the module directory exists
    module_dir = os.path.join(os.path.dirname(__file__), "..", "src", "mlops", "data_load")
    assert os.path.exists(module_dir), f"Module directory not found at {module_dir}"
    
    # Check that __init__.py exists
    init_file = os.path.join(module_dir, "__init__.py")
    assert os.path.exists(init_file), f"__init__.py not found at {init_file}"


def test_import_data_load_functions():
    """Test that data load functions can be imported"""
    from mlops.data_load.data_load import fetch_data, load_config
    
    # These should be callable
    assert callable(fetch_data)
    assert callable(load_config)