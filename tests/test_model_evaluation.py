import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pickle
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from mlops.evaluation.evaluation import ModelEvaluator, generate_report


@pytest.fixture
def dummy_dataframe():
    """Fixture for a dummy DataFrame that includes all features expected
    by define_features_and_label"""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2024-01-01", periods=10),
            "BTCUSDT_price": np.linspace(100, 110, 10),
            "ETHUSDT_price": np.linspace(50, 60, 10),
            "BNBUSDT_price": np.linspace(30, 35, 10),
            "XRPUSDT_price": np.linspace(0.3, 0.35, 10),
            "ADAUSDT_price": np.linspace(0.2, 0.25, 10),
            "SOLUSDT_price": np.linspace(20, 22, 10),
            "ETHUSDT_funding_rate": np.random.uniform(0.01, 0.03, 10),
            "BNBUSDT_funding_rate": np.random.uniform(0.01, 0.03, 10),
            "XRPUSDT_funding_rate": np.random.uniform(0.01, 0.03, 10),
            "ADAUSDT_funding_rate": np.random.uniform(0.01, 0.03, 10),
            "SOLUSDT_funding_rate": np.random.uniform(0.01, 0.03, 10),
            "BTCUSDT_funding_rate": np.random.uniform(0.01, 0.03, 10),
        }
    )


@pytest.fixture
def evaluator():
    # Provide dummy paths and a minimal config
    dummy_model_path = "models/linear_regression.pkl"
    dummy_test_data_dir = "data/processed/"
    dummy_config = {"artifacts": {"metrics_path": "models/metrics.json"}, "model": {}}
    return ModelEvaluator(
        model_path=dummy_model_path,
        test_data_dir=dummy_test_data_dir,
        config=dummy_config,
    )


def test_plot_confusion_matrix(evaluator):
    """Test that the confusion matrix is plotted and saved correctly."""
    y_test = pd.Series([0, 1, 0, 1])
    predictions = np.array([0, 1, 1, 0])

    with mock.patch("mlops.evaluation.evaluation.plt.savefig") as mock_save:
        evaluator.plot_confusion_matrix(y_test, predictions)
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        assert "confusion_matrix.png" in args[0]


def test_plot_regression_predictions(evaluator):
    """Test that the regression prediction plot is generated and saved."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4),
            "BTCUSDT_price": [100, 101, 102, 103],
        }
    )
    y_true = pd.Series([100, 101, 102, 103])
    y_pred = np.array([100.5, 100.8, 102.2, 102.7])

    with mock.patch("mlops.evaluation.evaluation.plt.savefig") as mock_save:
        evaluator.plot_regression_predictions(df, y_true, y_pred)
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        assert "price_prediction_plot.png" in args[0]


def test_save_metrics_report(tmp_path, evaluator):
    """Test saving regression and classification metrics to a JSON file."""
    regression_metrics = {"RMSE": 1.23}
    classification_metrics = {"Accuracy": 0.9, "F1 Score": 0.89, "ROC AUC": 0.88}

    metrics_path = tmp_path / "metrics.json"
    evaluator.config["artifacts"]["metrics_path"] = str(metrics_path)

    evaluator.save_metrics_report(regression_metrics, classification_metrics)

    assert metrics_path.exists()
    with open(metrics_path, "r") as f:
        data = json.load(f)
        assert "linear_regression" in data
        assert "logistic_regression" in data


def test_generate_report_calls_evaluate_models():
    """Ensure that generate_report calls evaluate_models as expected."""
    with mock.patch("mlops.evaluation.evaluation.evaluate_models") as mock_eval:
        generate_report({"some": "config"})
        mock_eval.assert_called_once()


def test_load_model_file_not_found(evaluator):
    """Test that loading a missing model raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        evaluator.load_model("non_existent_model.pkl")


def test_load_model_success(tmp_path, evaluator):
    """Test successful loading of a pickled model."""
    dummy_model = {"model": "dummy"}
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(dummy_model, f)

    result = evaluator.load_model(str(model_path))
    assert result == dummy_model


def test_load_both_models(tmp_path, evaluator):
    """Test that both regression and classification models are loaded."""
    dummy_model = {"name": "model"}
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    linear_path = model_dir / "linear_regression.pkl"
    logistic_path = model_dir / "logistic_regression.pkl"

    for path in [linear_path, logistic_path]:
        with open(path, "wb") as f:
            pickle.dump(dummy_model, f)

    evaluator.config["model"] = {
        "linear_regression": {"save_path": str(linear_path)},
        "logistic_regression": {"save_path": str(logistic_path)},
    }

    price_model, direction_model = evaluator.load_both_models()
    assert price_model == dummy_model
    assert direction_model == dummy_model
