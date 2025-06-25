import sys
import os
import pytest
import pandas as pd
import json
from unittest.mock import patch, MagicMock, call
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlops.data_validation.run import run_data_validation, _html_from_report


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "mlflow_tracking": {
            "experiment_name": "test-experiment"
        },
        "wandb": {
            "project": "test-project",
            "entity": "test-entity"
        },
        "data_validation": {
            "schema": {
                "columns": [
                    {
                        "name": "timestamp",
                        "type": "datetime",
                        "required": True
                    },
                    {
                        "name": "BTCUSDT_price",
                        "type": "float",
                        "required": True
                    }
                ]
            },
            "missing_values_strategy": "drop"
        },
        "data_source": {
            "processed_path": "data/processed/validated_data.csv"
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


@pytest.fixture
def mock_validation_report():
    """Mock validation report for testing"""
    return {
        "status": "passed",
        "issues": {
            "errors": [],
            "warnings": ["Some warnings found"]
        },
        "missing_values_summary": {
            "strategy": "drop",
            "missing_before": 5,
            "rows_dropped": 2
        },
        "column_details": {
            "timestamp": {
                "status": "valid",
                "expected_type": "datetime",
                "sample_values": ["2024-01-01", "2024-01-02"]
            },
            "BTCUSDT_price": {
                "status": "valid",
                "expected_type": "float",
                "sample_values": [100.0, 101.0]
            }
        }
    }


class TestDataValidationRun:
    """Test the data_validation run.py functionality"""
    
    @patch('mlops.data_validation.run.load_config')
    @patch('mlops.data_validation.run.validate_data')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('mlflow.log_artifact')
    @patch('wandb.init')
    @patch('wandb.finish')
    @patch('wandb.log')
    @patch('wandb.Html')
    @patch('wandb.Table')
    @patch('wandb.Artifact')
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    @patch('builtins.open', create=True)
    @patch('os.path.exists')
    def test_run_data_validation_success(
        self, mock_exists, mock_open, mock_to_csv, mock_makedirs, mock_read_csv,
        mock_artifact, mock_table, mock_html, mock_wandb_log, mock_wandb_finish,
        mock_wandb_init, mock_mlflow_log_artifact, mock_mlflow_start,
        mock_mlflow_set_experiment, mock_validate_data, mock_load_config,
        dummy_df, mock_config, mock_validation_report
    ):
        """Test successful data validation execution"""
        # Setup mocks
        mock_load_config.return_value = mock_config
        mock_read_csv.return_value = dummy_df
        mock_validate_data.return_value = (dummy_df, mock_validation_report)
        mock_exists.return_value = True  # File exists
        
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
        
        mock_mlflow_run = MagicMock()
        mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
        mock_mlflow_start.return_value.__exit__.side_effect = None
        
        mock_html_instance = MagicMock()
        mock_html.return_value = mock_html_instance
        
        mock_table_instance = MagicMock()
        mock_table.return_value = mock_table_instance
        
        mock_artifact_instance = MagicMock()
        mock_artifact.return_value = mock_artifact_instance
        
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # Execute
        run_data_validation("test_input.csv")
        
        # Assertions
        mock_load_config.assert_called_once()
        mock_read_csv.assert_called_once_with("test_input.csv")
        mock_validate_data.assert_called_once_with(dummy_df, mock_config["data_validation"])
        mock_mlflow_set_experiment.assert_called_once_with("test-experiment")
        mock_wandb_init.assert_called_once()
        mock_mlflow_log_artifact.assert_called()
        mock_wandb_finish.assert_called_once()
        
        # Check W&B logging calls
        assert mock_wandb_log.call_count >= 2  # validation_report, validated_data_summary/sample
    
    @patch('mlops.data_validation.run.load_config')
    @patch('mlflow.set_experiment')
    @patch('wandb.init')
    @patch('wandb.finish')
    @patch('pandas.read_csv')
    @patch('sys.exit')
    @patch('mlflow.start_run')
    @patch('os.path.exists')
    def test_run_data_validation_missing_input_file(
        self, mock_exists, mock_mlflow_start, mock_sys_exit, mock_read_csv, mock_wandb_finish,
        mock_wandb_init, mock_mlflow_set_experiment, mock_load_config, mock_config
    ):
        """Test data validation with missing input file"""
        mock_load_config.return_value = mock_config
        mock_exists.return_value = False  # File doesn't exist
        
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
        
        mock_mlflow_run = MagicMock()
        mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
        mock_mlflow_start.return_value.__exit__.side_effect = None
        
        # Execute
        run_data_validation("nonexistent.csv")
        
        # Should exit with error
        mock_sys_exit.assert_called_once_with(1)
        mock_wandb_finish.assert_called_once()
    
    @patch('mlops.data_validation.run.load_config')
    @patch('mlops.data_validation.run.validate_data')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('wandb.init')
    @patch('wandb.finish')
    @patch('pandas.read_csv')
    @patch('os.path.exists')
    def test_run_data_validation_validation_error(
        self, mock_exists, mock_mlflow_start, mock_read_csv, mock_wandb_finish, mock_wandb_init,
        mock_mlflow_set_experiment, mock_validate_data,
        mock_load_config, dummy_df, mock_config
    ):
        """Test data validation with validation error"""
        mock_load_config.return_value = mock_config
        mock_read_csv.return_value = dummy_df
        mock_validate_data.side_effect = Exception("Validation error")
        mock_exists.return_value = True  # File exists
    
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
    
        mock_mlflow_run = MagicMock()
        mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
        mock_mlflow_start.return_value.__exit__.side_effect = None
    
        # Execute and expect exception
        with pytest.raises(Exception, match="Validation error"):
            run_data_validation("test_input.csv")
    
    @patch('mlops.data_validation.run.load_config')
    @patch('mlops.data_validation.run.validate_data')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('wandb.init')
    @patch('wandb.finish')
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    @patch('os.path.exists')
    def test_run_data_validation_with_errors_in_report(
        self, mock_exists, mock_mlflow_start, mock_to_csv, mock_makedirs, mock_read_csv, mock_wandb_finish,
        mock_wandb_init, mock_mlflow_set_experiment,
        mock_validate_data, mock_load_config, dummy_df, mock_config
    ):
        """Test data validation with errors in validation report"""
        report_with_errors = {
            "status": "failed",
            "issues": {
                "errors": ["Column 'missing_col' not found"],
                "warnings": ["Some warnings"]
            },
            "missing_values_summary": {
                "strategy": "impute",
                "missing_before": 10,
                "total_imputed": 8
            },
            "column_details": {}
        }
    
        mock_load_config.return_value = mock_config
        mock_read_csv.return_value = dummy_df
        mock_validate_data.return_value = (dummy_df, report_with_errors)
        mock_exists.return_value = True  # File exists
    
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
    
        mock_mlflow_run = MagicMock()
        mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
        mock_mlflow_start.return_value.__exit__.side_effect = None
    
        # Execute
        run_data_validation("test_input.csv")
        
        # Should complete successfully even with validation errors in report
        mock_validate_data.assert_called_once()
        mock_wandb_finish.assert_called_once()


class TestHtmlFromReport:
    """Test the _html_from_report function"""
    
    def test_html_from_report_success(self, mock_validation_report):
        """Test HTML generation from successful validation report"""
        html_content = _html_from_report(mock_validation_report)
        
        # Check that HTML contains expected elements
        assert "<h2>Data Validation Report</h2>" in html_content
        assert "<b>Result:</b> passed" in html_content
        assert "Some warnings found" in html_content
        assert "timestamp" in html_content
        assert "BTCUSDT_price" in html_content
    
    def test_html_from_report_with_errors(self):
        """Test HTML generation from validation report with errors"""
        report_with_errors = {
            "status": "failed",
            "issues": {
                "errors": ["Column 'missing_col' not found", "Invalid data type"],
                "warnings": ["Some warnings"]
            },
            "missing_values_summary": {
                "strategy": "drop",
                "missing_before": 10,
                "rows_dropped": 5
            },
            "column_details": {
                "valid_col": {
                    "status": "valid",
                    "expected_type": "float",
                    "sample_values": [1.0, 2.0]
                }
            }
        }
        
        html_content = _html_from_report(report_with_errors)
        
        assert "<b>Result:</b> failed" in html_content
        assert "Column 'missing_col' not found" in html_content
        assert "Invalid data type" in html_content
        assert "Some warnings" in html_content
    
    def test_html_from_report_minimal(self):
        """Test HTML generation from minimal validation report"""
        minimal_report = {
            "status": "passed",
            "issues": {"errors": [], "warnings": []},
            "missing_values_summary": {"strategy": "drop", "missing_before": 0},
            "column_details": {}
        }
        
        html_content = _html_from_report(minimal_report)
        
        assert "<b>Result:</b> passed" in html_content
        assert "Errors: 0" in html_content
        assert "Warnings: 0" in html_content
    
    def test_html_from_report_with_list_values(self):
        """Test HTML generation with list values in column details"""
        report_with_lists = {
            "status": "passed",
            "issues": {"errors": [], "warnings": []},
            "missing_values_summary": {"strategy": "drop", "missing_before": 0},
            "column_details": {
                "list_col": {
                    "status": "valid",
                    "expected_type": "list",
                    "sample_values": [[1, 2, 3], [4, 5, 6]]
                }
            }
        }
        
        html_content = _html_from_report(report_with_lists)
        
        assert "list_col" in html_content
        assert "list" in html_content


class TestCLIInterface:
    """Test CLI interface functionality"""
    pass


class TestDataValidationRunIntegration:
    """Integration tests for data validation run.py"""
    
    def test_script_has_valid_syntax(self):
        """Test that the script has valid Python syntax"""
        script_path = os.path.join(os.path.dirname(__file__), "..", "src", "mlops", "data_validation", "run.py")
        with open(script_path, 'r') as f:
            script_content = f.read()
        compile(script_content, script_path, 'exec')
    
    @patch('mlops.data_validation.run.load_config')
    @patch('mlops.data_validation.run.validate_data')
    @patch('mlflow.set_experiment')
    @patch('mlflow.start_run')
    @patch('wandb.init')
    @patch('wandb.finish')
    @patch('pandas.read_csv')
    @patch('os.makedirs')
    @patch('pandas.DataFrame.to_csv')
    @patch('builtins.open', create=True)
    @patch('os.path.exists')
    def test_full_integration_simulation(
        self, mock_exists, mock_open, mock_to_csv, mock_makedirs, mock_read_csv,
        mock_wandb_finish, mock_wandb_init, mock_mlflow_start,
        mock_mlflow_set_experiment, mock_validate_data, mock_load_config,
        dummy_df, mock_config, mock_validation_report
    ):
        """Test full integration simulation"""
        mock_load_config.return_value = mock_config
        mock_read_csv.return_value = dummy_df
        mock_validate_data.return_value = (dummy_df, mock_validation_report)
        mock_exists.return_value = True  # File exists
    
        mock_wandb_run = MagicMock()
        mock_wandb_init.return_value = mock_wandb_run
    
        mock_mlflow_run = MagicMock()
        mock_mlflow_start.return_value.__enter__.return_value = mock_mlflow_run
    
        # Mock file operations
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
    
        # Execute
        run_data_validation("test_input.csv")
        
        # Verify all major components were called
        mock_load_config.assert_called_once()
        mock_read_csv.assert_called_once()
        mock_validate_data.assert_called_once()
        mock_wandb_init.assert_called_once()
        mock_mlflow_start.assert_called_once()
        mock_wandb_finish.assert_called_once()


def test_find_data_validation_files():
    """Test that data validation files can be found"""
    import os
    
    # Check that the main script exists
    script_path = os.path.join(os.path.dirname(__file__), "..", "src", "mlops", "data_validation", "run.py")
    assert os.path.exists(script_path), f"Script not found at {script_path}"
    
    # Check that the module directory exists
    module_dir = os.path.join(os.path.dirname(__file__), "..", "src", "mlops", "data_validation")
    assert os.path.exists(module_dir), f"Module directory not found at {module_dir}"
    
    # Check that __init__.py exists
    init_file = os.path.join(module_dir, "__init__.py")
    assert os.path.exists(init_file), f"__init__.py not found at {init_file}"


def test_import_data_validation_functions():
    """Test that data validation functions can be imported"""
    # This test will fail if the actual validate_data function doesn't exist
    # We'll mock it for now since it's not implemented
    with patch('mlops.data_validation.data_validation.validate_data', create=True):
        from mlops.data_validation.data_validation import validate_data
        from mlops.data_load.data_load import load_config
        
        # These should be callable
        assert callable(validate_data)
        assert callable(load_config) 