import sys
import os
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, call
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mlops import main


@pytest.fixture
def dummy_df():
    """Create dummy input dataframe for testing"""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=5),
            "BTCUSDT_price": [100, 101, 102, 103, 104],
            "ETHUSDT_price": [50, 51, 52, 53, 54],
            "BTCUSDT_funding_rate": [0.01, 0.02, 0.015, 0.017, 0.018],
            "price_direction": [1, 0, 1, 1, 0],
        }
    )


@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
            "log_file": "logs/test.log"
        },
        "mlflow_tracking": {
            "experiment_name": "test-experiment"
        },
        "wandb": {
            "project": "test-project",
            "entity": "test-entity"
        },
        "data_validation": {
            "schema": {"columns": []},
            "missing_values_strategy": "drop"
        },
        "data_source": {
            "raw_path": "data/raw/raw_data.csv",
            "processed_path": "data/processed/processed_data.csv"
        },
        "symbols": ["BTCUSDT", "ETHUSDT"]
    }


class TestSetupLogger:
    """Test the setup_logger function"""
    
    @patch('mlops.main.load_config')
    @patch('os.makedirs')
    @patch('logging.basicConfig')
    @patch('logging.getLogger')
    def test_setup_logger_success(self, mock_get_logger, mock_basic_config, mock_makedirs, mock_load_config, mock_config):
        """Test successful logger setup"""
        mock_load_config.return_value = mock_config
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        main.setup_logger()
        
        mock_load_config.assert_called_once_with("conf/config.yaml")
        mock_basic_config.assert_called_once()
        mock_get_logger.assert_called_with("")
        mock_logger.addHandler.assert_called_once()
    
    @patch('mlops.main.load_config')
    @patch('logging.basicConfig')
    def test_setup_logger_without_log_file(self, mock_basic_config, mock_load_config):
        """Test logger setup without log file"""
        config_without_log_file = {
            "logging": {
                "level": "DEBUG",
                "format": "%(levelname)s - %(message)s"
            }
        }
        mock_load_config.return_value = config_without_log_file
        
        main.setup_logger()
        
        mock_basic_config.assert_called_once()


class TestRunFullPipeline:
    """Test the run_full_pipeline function"""
    pass


class TestPreprocessData:
    """Test the preprocess_data function"""
    
    @patch('mlops.main.load_config')
    @patch('mlops.main.scale_features')
    @patch('mlops.main.smote_oversample')
    def test_preprocess_data_success(self, mock_smote, mock_scale, mock_load_config, dummy_df):
        """Test successful preprocessing"""
        mock_load_config.return_value = {}
        mock_scale.return_value = (pd.DataFrame(), pd.DataFrame(), MagicMock())
        mock_smote.return_value = (pd.DataFrame(), pd.Series())
        
        feature_cols = ["BTCUSDT_price", "ETHUSDT_price"]
        y_class = pd.Series([1, 0, 1, 0, 1])
        
        result = main.preprocess_data(dummy_df, feature_cols, y_class)
        
        mock_scale.assert_called_once_with(dummy_df, feature_cols)
        mock_smote.assert_called_once()
        assert len(result) == 2


class TestRunUntilFeatureEngineering:
    """Test the run_until_feature_engineering function"""
    
    @patch('mlops.main.fetch_data')
    @patch('mlops.main.load_config')
    @patch('mlops.main.validate_data')
    @patch('mlops.main.define_features_and_label')
    @patch('mlops.main.create_price_direction_label')
    @patch('mlops.main.prepare_features')
    def test_run_until_feature_engineering_success(
        self, mock_prepare_features, mock_create_label, mock_define_features,
        mock_validate_data, mock_load_config, mock_fetch_data, dummy_df, mock_config
    ):
        """Test successful feature engineering execution"""
        mock_load_config.return_value = mock_config
        mock_fetch_data.return_value = dummy_df
        mock_validate_data.return_value = dummy_df
        mock_define_features.return_value = (["BTCUSDT_price"], "BTCUSDT_price")
        mock_create_label.return_value = dummy_df
        mock_prepare_features.return_value = (dummy_df, pd.Series(), pd.Series())
        
        result = main.run_until_feature_engineering()
        
        assert len(result) == 4  # X, y_reg, y_class, df
        mock_fetch_data.assert_called_once()
        mock_validate_data.assert_called_once()
        mock_define_features.assert_called_once()
        mock_create_label.assert_called_once()
        mock_prepare_features.assert_called_once()


class TestMainFunction:
    """Test the main function"""
    pass
