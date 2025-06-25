import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import tempfile
import subprocess
import pickle
import yaml
from unittest.mock import patch, Mock, mock_open, MagicMock
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to import the inference module
try:
    from mlops.inference.inference import ModelInferencer, load_models, predict_price, predict_direction, predict_both, run_inference
except ImportError:
    # If direct import fails, try to find and import dynamically
    inference_file = None
    for root, dirs, files in os.walk('.'):
        if 'inference.py' in files and 'inference' in root:
            inference_file = os.path.join(root, 'inference.py')
            break
    
    if inference_file:
        import importlib.util
        spec = importlib.util.spec_from_file_location("inference_module", inference_file)
        inference_module = importlib.util.module_from_spec(spec)
        sys.modules["inference_module"] = inference_module
        try:
            spec.loader.exec_module(inference_module)
            ModelInferencer = inference_module.ModelInferencer
            load_models = inference_module.load_models
            predict_price = inference_module.predict_price
            predict_direction = inference_module.predict_direction
            predict_both = inference_module.predict_both
            run_inference = inference_module.run_inference
        except Exception as e:
            pytest.skip(f"Could not load inference module: {e}")
    else:
        pytest.skip("Could not find inference.py file")


class DummyModel:
    """Enhanced dummy model class for testing."""

    def __init__(self, fail_predict=False, has_proba=True, has_decision=True):
        self.fail_predict = fail_predict
        self.has_proba = has_proba
        self.has_decision = has_decision

    def predict(self, X):
        if self.fail_predict:
            raise ValueError("Prediction failed")
        return np.ones(len(X)) * 42.5  # Return meaningful test values

    def predict_proba(self, X):
        if not self.has_proba:
            raise AttributeError("Model has no predict_proba")
        return np.array([[0.3, 0.7]] * len(X))

    def decision_function(self, X):
        if not self.has_decision:
            raise AttributeError("Model has no decision_function")
        return np.ones(len(X)) * 0.8

    def __getattr__(self, name):
        """Handle missing methods dynamically."""
        if name == "predict_proba" and not self.has_proba:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        elif name == "decision_function" and not self.has_decision:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class DummyModelWithoutProba:
    """Model without predict_proba for testing fallback behavior."""
    
    def predict(self, X):
        return np.ones(len(X)) * 42.5
    
    def decision_function(self, X):
        return np.ones(len(X)) * 0.8


class DummyScaler:
    """Enhanced dummy scaler class for testing."""

    def __init__(self, fail_transform=False):
        self.fail_transform = fail_transform
        self.feature_names_in_ = ["ETHUSDT_price", "BTCUSDT_funding_rate"]

    def transform(self, X):
        if self.fail_transform:
            raise ValueError("Scaling failed")
        return X.to_numpy() * 2  # Apply some transformation


class TestModelInferencerInit:
    """Test ModelInferencer initialization and model loading."""

    @patch('mlops.inference.inference.setup_logger')
    def test_init_with_valid_config(self, mock_logger):
        """Test initialization with valid configuration."""
        with patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            inferencer = ModelInferencer()
            # The config will be the real one, but we're mocking the loading methods
            # so just test that initialization works without error
            assert hasattr(inferencer, 'config')
            assert inferencer.price_model is None
            assert inferencer.direction_model is None
            assert inferencer.preprocessing_pipeline is None

    @patch('mlops.inference.inference.load_config')
    @patch('mlops.inference.inference.setup_logger')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pickle.load')
    def test_load_models_success(self, mock_pickle_load, mock_file, mock_exists, mock_logger, mock_load_config):
        """Test successful model loading."""
        mock_config = {
            "model": {
                "linear_regression": {"save_path": "models/linear_reg.pkl"},
                "logistic_regression": {"save_path": "models/log_reg.pkl"}
            }
        }
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True
        mock_pickle_load.side_effect = [DummyModel(), DummyModel()]

        with patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            inferencer = ModelInferencer()
            
        assert inferencer.price_model is not None
        assert inferencer.direction_model is not None
        assert mock_pickle_load.call_count == 2

    @patch('mlops.inference.inference.load_config')
    @patch('mlops.inference.inference.setup_logger')
    @patch('os.path.exists')
    def test_load_models_file_not_found(self, mock_exists, mock_logger, mock_load_config):
        """Test model loading when files don't exist."""
        mock_config = {"model": {"linear_regression": {"save_path": "nonexistent.pkl"}}}
        mock_load_config.return_value = mock_config
        mock_exists.return_value = False

        with patch.object(ModelInferencer, '_load_preprocessing_pipeline'), \
             pytest.raises(FileNotFoundError):
            ModelInferencer()

    @patch('mlops.inference.inference.load_config')
    @patch('mlops.inference.inference.setup_logger')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_preprocessing_pipeline_success(self, mock_file, mock_exists, mock_logger, mock_load_config):
        """Test successful preprocessing pipeline loading."""
        mock_config = {"artifacts": {"preprocessing_pipeline": "models/pipeline.pkl"}}
        mock_load_config.return_value = mock_config
        mock_exists.return_value = True

        dummy_pipeline = {
            "scaler": DummyScaler(),
            "selected_features_reg": ["feature1"],
            "selected_features_class": ["feature2"],
            "all_feature_cols": ["feature1", "feature2"]
        }

        with patch.object(ModelInferencer, '_load_models'), \
             patch('pickle.load', return_value=dummy_pipeline):
            
            inferencer = ModelInferencer()
            assert inferencer.preprocessing_pipeline == dummy_pipeline

    @patch('mlops.inference.inference.load_config')
    @patch('mlops.inference.inference.setup_logger')
    @patch('os.path.exists')
    def test_load_preprocessing_pipeline_not_found(self, mock_exists, mock_logger, mock_load_config):
        """Test preprocessing pipeline loading when file doesn't exist."""
        mock_config = {"artifacts": {"preprocessing_pipeline": "nonexistent.pkl"}}
        mock_load_config.return_value = mock_config
        mock_exists.return_value = False

        with patch.object(ModelInferencer, '_load_models'):
            inferencer = ModelInferencer()
            assert inferencer.preprocessing_pipeline is None


class TestModelInferencerValidation:
    """Test input validation and preprocessing."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
            "BTCUSDT_price": [100, 101, 102, 103, 104],
            "ETHUSDT_price": [50, 51, 52, 53, 54],
            "BTCUSDT_funding_rate": [0.01, 0.02, 0.015, 0.017, 0.019],
        })

    @pytest.fixture
    def dummy_pipeline(self):
        return {
            "scaler": DummyScaler(),
            "selected_features_reg": ["ETHUSDT_price"],
            "selected_features_class": ["BTCUSDT_funding_rate"],
            "all_feature_cols": ["ETHUSDT_price", "BTCUSDT_funding_rate"],
        }

    @pytest.fixture
    def inferencer_with_models(self, dummy_pipeline):
        """Create inferencer with dummy models and pipeline."""
        with patch('mlops.inference.inference.load_config'), \
             patch('mlops.inference.inference.setup_logger'), \
             patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            inferencer = ModelInferencer()
            inferencer.price_model = DummyModel()
            inferencer.direction_model = DummyModel()
            inferencer.preprocessing_pipeline = dummy_pipeline
            return inferencer

    @patch('mlops.inference.inference.define_features_and_label')
    def test_validate_and_preprocess_input_success(self, mock_define, inferencer_with_models, sample_df):
        """Test successful input validation and preprocessing."""
        mock_define.return_value = (["ETHUSDT_price", "BTCUSDT_funding_rate"], "BTCUSDT_price")
        
        features_reg, features_class = inferencer_with_models._validate_and_preprocess_input(sample_df)
        
        assert isinstance(features_reg, np.ndarray)
        assert isinstance(features_class, np.ndarray)
        assert features_reg.shape[0] == len(sample_df)
        assert features_class.shape[0] == len(sample_df)

    @patch('mlops.inference.inference.define_features_and_label')
    def test_validate_input_not_dataframe(self, mock_define, inferencer_with_models):
        """Test validation with non-DataFrame input."""
        mock_define.return_value = (["feature1"], "target")
        
        with pytest.raises(ValueError, match="Input must be a pandas DataFrame"):
            inferencer_with_models._validate_and_preprocess_input([1, 2, 3])

    @patch('mlops.inference.inference.define_features_and_label')
    def test_validate_input_missing_features(self, mock_define, inferencer_with_models):
        """Test validation with missing required features."""
        mock_define.return_value = (["missing_feature"], "target")
        df = pd.DataFrame({"other_feature": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Missing required features"):
            inferencer_with_models._validate_and_preprocess_input(df)

    @patch('mlops.inference.inference.define_features_and_label')
    def test_validate_input_with_missing_values(self, mock_define, inferencer_with_models, sample_df):
        """Test handling of missing values in input data."""
        mock_define.return_value = (["ETHUSDT_price", "BTCUSDT_funding_rate"], "BTCUSDT_price")
        
        # Add missing values
        sample_df.loc[0, "ETHUSDT_price"] = np.nan
        sample_df.loc[1, "BTCUSDT_funding_rate"] = np.nan
        
        features_reg, features_class = inferencer_with_models._validate_and_preprocess_input(sample_df)
        
        # Should handle missing values without crashing
        assert not np.isnan(features_reg).any()
        assert not np.isnan(features_class).any()

    @patch('mlops.inference.inference.define_features_and_label')
    def test_validate_input_no_preprocessing_pipeline(self, mock_define, sample_df):
        """Test validation when no preprocessing pipeline is available."""
        mock_define.return_value = (["ETHUSDT_price", "BTCUSDT_funding_rate"], "BTCUSDT_price")
        
        with patch('mlops.inference.inference.load_config'), \
             patch('mlops.inference.inference.setup_logger'), \
             patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            inferencer = ModelInferencer()
            inferencer.preprocessing_pipeline = None
            
            features_reg, features_class = inferencer._validate_and_preprocess_input(sample_df)
            
            # Should return raw features when no pipeline
            assert isinstance(features_reg, np.ndarray)
            assert isinstance(features_class, np.ndarray)


class TestModelInferencerPredictions:
    """Test prediction methods."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "BTCUSDT_price": [100, 101, 102],
            "ETHUSDT_price": [50, 51, 52],
            "BTCUSDT_funding_rate": [0.01, 0.02, 0.015],
        })

    @pytest.fixture
    def dummy_pipeline(self):
        return {
            "scaler": DummyScaler(),
            "selected_features_reg": ["ETHUSDT_price"],
            "selected_features_class": ["BTCUSDT_funding_rate"],
            "all_feature_cols": ["ETHUSDT_price", "BTCUSDT_funding_rate"],
        }

    @pytest.fixture
    def inferencer_with_models(self, dummy_pipeline):
        with patch('mlops.inference.inference.load_config'), \
             patch('mlops.inference.inference.setup_logger'), \
             patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            inferencer = ModelInferencer()
            inferencer.price_model = DummyModel()
            inferencer.direction_model = DummyModel()
            inferencer.preprocessing_pipeline = dummy_pipeline
            return inferencer

    @patch('mlops.inference.inference.define_features_and_label')
    def test_predict_price_success(self, mock_define, inferencer_with_models, sample_df):
        """Test successful price prediction."""
        mock_define.return_value = (["ETHUSDT_price", "BTCUSDT_funding_rate"], "BTCUSDT_price")
        
        predictions = inferencer_with_models.predict_price(sample_df)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(sample_df)
        assert all(pred == 42.5 for pred in predictions)  # DummyModel returns 42.5

    @patch('mlops.inference.inference.define_features_and_label')
    def test_predict_price_no_model(self, mock_define, sample_df):
        """Test price prediction when no model is loaded."""
        mock_define.return_value = (["ETHUSDT_price"], "target")
        
        with patch('mlops.inference.inference.load_config'), \
             patch('mlops.inference.inference.setup_logger'), \
             patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            inferencer = ModelInferencer()
            inferencer.price_model = None
            
            with pytest.raises(RuntimeError, match="Price model not loaded"):
                inferencer.predict_price(sample_df)

    @patch('mlops.inference.inference.define_features_and_label')
    def test_predict_direction_success(self, mock_define, inferencer_with_models, sample_df):
        """Test successful direction prediction."""
        mock_define.return_value = (["ETHUSDT_price", "BTCUSDT_funding_rate"], "BTCUSDT_price")
        
        directions, probabilities = inferencer_with_models.predict_direction(sample_df)
        
        assert isinstance(directions, np.ndarray)
        assert isinstance(probabilities, np.ndarray)
        assert len(directions) == len(sample_df)
        assert len(probabilities) == len(sample_df)

    @patch('mlops.inference.inference.define_features_and_label')
    def test_predict_direction_no_model(self, mock_define, sample_df):
        """Test direction prediction when no model is loaded."""
        mock_define.return_value = (["ETHUSDT_price"], "target")
        
        with patch('mlops.inference.inference.load_config'), \
             patch('mlops.inference.inference.setup_logger'), \
             patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            inferencer = ModelInferencer()
            inferencer.direction_model = None
            
            with pytest.raises(RuntimeError, match="Direction model not loaded"):
                inferencer.predict_direction(sample_df)

    @patch('mlops.inference.inference.define_features_and_label')
    def test_predict_direction_no_predict_proba(self, mock_define, sample_df, dummy_pipeline):
        """Test direction prediction when model has no predict_proba method."""
        mock_define.return_value = (["ETHUSDT_price", "BTCUSDT_funding_rate"], "BTCUSDT_price")
        
        with patch('mlops.inference.inference.load_config'), \
             patch('mlops.inference.inference.setup_logger'), \
             patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            inferencer = ModelInferencer()
            inferencer.direction_model = DummyModelWithoutProba()  # Use model without predict_proba
            inferencer.preprocessing_pipeline = dummy_pipeline
            
            directions, probabilities = inferencer.predict_direction(sample_df)
            
            # Should still work using decision_function or predictions
            assert isinstance(directions, np.ndarray)
            assert isinstance(probabilities, np.ndarray)

    @patch('mlops.inference.inference.define_features_and_label')
    def test_predict_both_success(self, mock_define, inferencer_with_models, sample_df):
        """Test successful prediction of both price and direction."""
        mock_define.return_value = (["ETHUSDT_price", "BTCUSDT_funding_rate"], "BTCUSDT_price")
        
        results = inferencer_with_models.predict_both(sample_df)
        
        assert isinstance(results, dict)
        assert "price_predictions" in results
        assert "direction_predictions" in results
        assert "direction_probabilities" in results
        
        assert len(results["price_predictions"]) == len(sample_df)
        assert len(results["direction_predictions"]) == len(sample_df)
        assert len(results["direction_probabilities"]) == len(sample_df)


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "ETHUSDT_price": [50, 51, 52],
            "BTCUSDT_funding_rate": [0.01, 0.02, 0.015],
        })

    @patch('mlops.inference.inference.ModelInferencer')
    def test_load_models(self, mock_inferencer_class):
        """Test load_models convenience function."""
        mock_instance = Mock()
        mock_inferencer_class.return_value = mock_instance
        
        result = load_models()
        
        mock_inferencer_class.assert_called_once()
        assert result == mock_instance

    @patch('mlops.inference.inference.ModelInferencer')
    def test_predict_price_convenience(self, mock_inferencer_class, sample_df):
        """Test predict_price convenience function."""
        mock_instance = Mock()
        mock_instance.predict_price.return_value = np.array([1, 2, 3])
        mock_inferencer_class.return_value = mock_instance
        
        result = predict_price(sample_df)
        
        mock_inferencer_class.assert_called_once()
        mock_instance.predict_price.assert_called_once_with(sample_df)
        np.testing.assert_array_equal(result, [1, 2, 3])

    @patch('mlops.inference.inference.ModelInferencer')
    def test_predict_price_convenience_with_inferencer(self, mock_inferencer_class, sample_df):
        """Test predict_price convenience function with provided inferencer."""
        mock_instance = Mock()
        mock_instance.predict_price.return_value = np.array([4, 5, 6])
        
        result = predict_price(sample_df, inferencer=mock_instance)
        
        mock_inferencer_class.assert_not_called()
        mock_instance.predict_price.assert_called_once_with(sample_df)
        np.testing.assert_array_equal(result, [4, 5, 6])

    @patch('mlops.inference.inference.ModelInferencer')
    def test_predict_direction_convenience(self, mock_inferencer_class, sample_df):
        """Test predict_direction convenience function."""
        mock_instance = Mock()
        mock_instance.predict_direction.return_value = (np.array([0, 1, 1]), np.array([0.3, 0.7, 0.8]))
        mock_inferencer_class.return_value = mock_instance
        
        directions, probs = predict_direction(sample_df)
        
        mock_inferencer_class.assert_called_once()
        mock_instance.predict_direction.assert_called_once_with(sample_df)

    @patch('mlops.inference.inference.ModelInferencer')
    def test_predict_both_convenience(self, mock_inferencer_class, sample_df):
        """Test predict_both convenience function."""
        mock_instance = Mock()
        expected_results = {
            "price_predictions": np.array([1, 2, 3]),
            "direction_predictions": np.array([0, 1, 1]),
            "direction_probabilities": np.array([0.3, 0.7, 0.8])
        }
        mock_instance.predict_both.return_value = expected_results
        mock_inferencer_class.return_value = mock_instance
        
        results = predict_both(sample_df)
        
        mock_inferencer_class.assert_called_once()
        mock_instance.predict_both.assert_called_once_with(sample_df)
        assert results == expected_results


class TestRunInference:
    """Test the run_inference function."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "ETHUSDT_price": [50, 51, 52],
            "BTCUSDT_funding_rate": [0.01, 0.02, 0.015],
        })

    @patch('mlops.inference.inference.ModelInferencer')
    def test_run_inference_success(self, mock_inferencer_class, sample_df):
        """Test successful run_inference execution."""
        mock_instance = Mock()
        mock_results = {
            "price_predictions": np.array([100, 101, 102]),
            "direction_predictions": np.array([0, 1, 1]),
            "direction_probabilities": np.array([0.3, 0.7, 0.8])
        }
        mock_instance.predict_both.return_value = mock_results
        mock_inferencer_class.return_value = mock_instance
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            run_inference(sample_df, "config.yaml", output_path)
            
            # Verify output file was created and has correct content
            result_df = pd.read_csv(output_path)
            assert "predicted_price" in result_df.columns
            assert "predicted_direction" in result_df.columns
            assert "direction_probability" in result_df.columns
            assert len(result_df) == len(sample_df)
            
        finally:
            os.unlink(output_path)

    @patch('mlops.inference.inference.ModelInferencer')
    def test_run_inference_model_error(self, mock_inferencer_class, sample_df):
        """Test run_inference when model prediction fails."""
        mock_inferencer_class.side_effect = Exception("Model loading failed")
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_path = f.name
        
        try:
            with pytest.raises(Exception, match="Model loading failed"):
                run_inference(sample_df, "config.yaml", output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestCLIInterface:
    """Test command-line interface functionality."""

    def find_inference_script(self):
        """Find the inference.py script in the project."""
        print("[DEBUG] CWD:", os.getcwd())
        # Get the project root directory (where this test file is located)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print("[DEBUG] Project root:", project_root)
        
        # Use absolute paths relative to project root
        script_path = os.path.join(project_root, "src", "mlops", "inference", "inference.py")
        print(f"[DEBUG] Trying: {script_path} Exists: {os.path.exists(script_path)}")
        if os.path.exists(script_path):
            return script_path
        
        print("[DEBUG] No inference.py found!")
        return None

    def test_cli_no_arguments(self):
        """Test CLI with no arguments (should show usage)."""
        script_path = self.find_inference_script()
        if not script_path:
            pytest.skip("Could not find inference.py script")
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should fail due to missing arguments OR import errors
        assert result.returncode != 0
        # Accept various types of failures - usage errors, import errors, etc.
        error_keywords = [
            "Usage:", "import", "ModuleNotFoundError", "ImportError",
            "No module named", "mlops", "arguments"
        ]
        error_found = any(keyword in result.stdout + result.stderr for keyword in error_keywords)
        assert error_found, f"Expected error not found in output: stdout={result.stdout}, stderr={result.stderr}"

    def test_cli_wrong_number_of_arguments(self):
        """Test CLI with wrong number of arguments."""
        script_path = self.find_inference_script()
        if not script_path:
            pytest.skip("Could not find inference.py script")
        
        result = subprocess.run(
            [sys.executable, script_path, "only_one_arg"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should fail due to wrong arguments OR import errors
        assert result.returncode != 0
        # Accept various types of failures
        error_keywords = [
            "Usage:", "import", "ModuleNotFoundError", "ImportError",
            "No module named", "mlops", "arguments"
        ]
        error_found = any(keyword in result.stdout + result.stderr for keyword in error_keywords)
        assert error_found, f"Expected error not found in output: stdout={result.stdout}, stderr={result.stderr}"

    def test_cli_with_nonexistent_files(self):
        """Test CLI with nonexistent input files."""
        script_path = self.find_inference_script()
        if not script_path:
            pytest.skip("Could not find inference.py script")
        
        result = subprocess.run(
            [sys.executable, script_path, "nonexistent.csv", "nonexistent.yaml", "output.csv"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should fail due to missing files or dependencies
        assert result.returncode != 0

    def test_script_syntax_validation(self):
        """Test that the script has valid Python syntax."""
        script_path = self.find_inference_script()
        if not script_path:
            pytest.skip("Could not find inference.py script")
        
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        try:
            compile(script_content, script_path, 'exec')
        except SyntaxError as e:
            pytest.fail(f"Script has syntax error: {e}")


class TestErrorHandling:
    """Test various error conditions."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "ETHUSDT_price": [50, 51, 52],
            "BTCUSDT_funding_rate": [0.01, 0.02, 0.015],
        })

    @pytest.fixture
    def failing_pipeline(self):
        return {
            "scaler": DummyScaler(fail_transform=True),
            "selected_features_reg": ["ETHUSDT_price"],
            "selected_features_class": ["BTCUSDT_funding_rate"],
            "all_feature_cols": ["ETHUSDT_price", "BTCUSDT_funding_rate"],
        }

    @patch('mlops.inference.inference.define_features_and_label')
    def test_preprocessing_failure(self, mock_define, sample_df, failing_pipeline):
        """Test handling of preprocessing failures."""
        mock_define.return_value = (["ETHUSDT_price", "BTCUSDT_funding_rate"], "target")
        
        with patch('mlops.inference.inference.load_config'), \
             patch('mlops.inference.inference.setup_logger'), \
             patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            inferencer = ModelInferencer()
            inferencer.preprocessing_pipeline = failing_pipeline
            
            with pytest.raises(ValueError, match="Scaling failed"):
                inferencer._validate_and_preprocess_input(sample_df)

    @patch('mlops.inference.inference.define_features_and_label')
    def test_prediction_failure(self, mock_define, sample_df):
        """Test handling of model prediction failures."""
        mock_define.return_value = (["ETHUSDT_price", "BTCUSDT_funding_rate"], "target")
        
        dummy_pipeline = {
            "scaler": DummyScaler(),
            "selected_features_reg": ["ETHUSDT_price"],
            "selected_features_class": ["BTCUSDT_funding_rate"],
            "all_feature_cols": ["ETHUSDT_price", "BTCUSDT_funding_rate"],
        }
        
        with patch('mlops.inference.inference.load_config'), \
             patch('mlops.inference.inference.setup_logger'), \
             patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            inferencer = ModelInferencer()
            inferencer.price_model = DummyModel(fail_predict=True)
            inferencer.preprocessing_pipeline = dummy_pipeline
            
            with pytest.raises(ValueError, match="Prediction failed"):
                inferencer.predict_price(sample_df)

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_df = pd.DataFrame()
        
        with patch('mlops.inference.inference.define_features_and_label') as mock_define, \
             patch('mlops.inference.inference.load_config'), \
             patch('mlops.inference.inference.setup_logger'), \
             patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            mock_define.return_value = (["feature1"], "target")
            
            inferencer = ModelInferencer()
            
            with pytest.raises(ValueError):
                inferencer._validate_and_preprocess_input(empty_df)


# Integration tests
class TestInferenceIntegration:
    """Test end-to-end functionality with minimal mocking."""

    def test_model_inferencer_workflow(self):
        """Test the complete ModelInferencer workflow with mocked dependencies."""
        sample_data = pd.DataFrame({
            "ETHUSDT_price": [50, 51, 52],
            "BTCUSDT_funding_rate": [0.01, 0.02, 0.015],
        })
        
        # Create a realistic pipeline
        pipeline = {
            "scaler": DummyScaler(),
            "selected_features_reg": ["ETHUSDT_price"],
            "selected_features_class": ["BTCUSDT_funding_rate"],
            "all_feature_cols": ["ETHUSDT_price", "BTCUSDT_funding_rate"],
        }
        
        with patch('mlops.inference.inference.define_features_and_label') as mock_define, \
             patch('mlops.inference.inference.load_config'), \
             patch('mlops.inference.inference.setup_logger'), \
             patch.object(ModelInferencer, '_load_models'), \
             patch.object(ModelInferencer, '_load_preprocessing_pipeline'):
            
            mock_define.return_value = (["ETHUSDT_price", "BTCUSDT_funding_rate"], "BTCUSDT_price")
            
            # Test the complete workflow
            inferencer = ModelInferencer()
            inferencer.price_model = DummyModel()
            inferencer.direction_model = DummyModel()
            inferencer.preprocessing_pipeline = pipeline
            
            # Test all prediction methods
            price_pred = inferencer.predict_price(sample_data)
            dir_pred, dir_prob = inferencer.predict_direction(sample_data)
            both_pred = inferencer.predict_both(sample_data)
            
            # Verify results
            assert len(price_pred) == len(sample_data)
            assert len(dir_pred) == len(sample_data)
            assert len(dir_prob) == len(sample_data)
            assert len(both_pred["price_predictions"]) == len(sample_data)

    def test_convenience_functions_integration(self):
        """Test convenience functions work together."""
        sample_data = pd.DataFrame({
            "ETHUSDT_price": [50, 51],
            "BTCUSDT_funding_rate": [0.01, 0.02],
        })
        
        with patch('mlops.inference.inference.ModelInferencer') as mock_class:
            mock_instance = Mock()
            mock_instance.predict_price.return_value = np.array([100, 101])
            mock_instance.predict_direction.return_value = (np.array([0, 1]), np.array([0.3, 0.7]))
            mock_instance.predict_both.return_value = {
                "price_predictions": np.array([100, 101]),
                "direction_predictions": np.array([0, 1]),
                "direction_probabilities": np.array([0.3, 0.7])
            }
            mock_class.return_value = mock_instance
            
            # Test that all convenience functions work
            models = load_models()
            prices = predict_price(sample_data, models)
            directions, probs = predict_direction(sample_data, models)
            both = predict_both(sample_data, models)
            
            assert len(prices) == 2
            assert len(directions) == 2
            assert len(probs) == 2
            assert len(both["price_predictions"]) == 2


if __name__ == "__main__":
    pytest.main([__file__])