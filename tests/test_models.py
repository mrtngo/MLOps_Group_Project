# import pytest
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from unittest.mock import patch, MagicMock
# from mlops.models.models import ModelTrainer

# @pytest.fixture
# def dummy_config():
#     return {
#         "model": {
#             "linear_regression": {
#                 "params": {},
#                 "save_path": "models/test_lr.pkl"
#             }
#         },
#         "artifacts": {
#             "preprocessing_pipeline": "models/test_pipeline.pkl"
#         },
#         "target": "price"
#     }

# @patch("mlops.models.models.load_config")
# def test_train_linear_regression(mock_config):
#     mock_config.return_value = {
#         "model": {
#             "linear_regression": {
#                 "params": {"fit_intercept": True},
#                 "save_path": "models/test_lr.pkl"
#             }
#         },
#         "artifacts": {
#             "preprocessing_pipeline": "models/test_pipeline.pkl"
#         },
#         "target": "price"
#     }

#     trainer = ModelTrainer()

#     # Dummy training data
#     X = pd.DataFrame(np.array([[1], [2], [3], [4]]), columns=["feature1"])
#     y = pd.Series([10, 20, 30, 40])

#     model = trainer.train_linear_regression(X, y)

#     assert isinstance(model, LinearRegression)
#     assert hasattr(model, "predict")
#     preds = model.predict(X)
#     assert np.allclose(preds, [10, 20, 30, 40], atol=1e-1)


import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from mlops.models.models import ModelTrainer


@patch("mlops.models.models.load_config")
@patch("mlops.models.models.smote_oversample")
@patch("mlops.models.models.select_features")
@patch("mlops.models.models.scale_features")
@patch("mlops.models.models.split_data")
@patch("mlops.models.models.prepare_features")
@patch("mlops.models.models.create_price_direction_label")
@patch("mlops.models.models.define_features_and_label")
def test_prepare_data(
    mock_define, mock_create_label, mock_prepare_feat, mock_split, mock_scale,
    mock_select, mock_smote, mock_config
):
    mock_config.return_value = {
        "model": {},
        "target": "price",
        "artifacts": {"preprocessing_pipeline": "models/test_pipeline.pkl"}
    }

    mock_define.return_value = (["f1", "f2"], "price")
    mock_create_label.return_value = pd.DataFrame({"f1": [1], "f2": [2], "price": [10], "price_direction": [1]})
    mock_prepare_feat.return_value = (np.array([[1, 2]]), pd.Series([10]), pd.Series([1]))
    mock_split.return_value = (np.array([[1, 2]]), np.array([[3, 4]]), pd.Series([10]), pd.Series([20]))

    scaler = StandardScaler().fit([[1, 2], [3, 4]])
    mock_scale.return_value = (np.array([[1.1, 2.2]]), np.array([[1.3, 2.5]]), scaler)

    mock_select.side_effect = lambda df, f: f
    mock_smote.return_value = (np.array([[9, 9]]), pd.Series([1]))

    trainer = ModelTrainer()
    df = pd.DataFrame({"f1": [1], "f2": [2], "price": [10]})
    result = trainer.prepare_data(df)

    assert isinstance(result, tuple)
    assert len(result) == 6


@patch("mlops.models.models.load_config")
def test_train_logistic_regression_runs(mock_config, tmp_path):
    mock_config.return_value = {
        "model": {},
        "artifacts": {"preprocessing_pipeline": "models/pipe.pkl"},
        "target": "price"
    }

    trainer = ModelTrainer()
    trainer.model_config = {
        "logistic_regression": {
            "params": {},
            "save_path": str(tmp_path / "logistic.pkl")
        }
    }

    X = pd.DataFrame([[0, 1], [1, 0]])
    y = pd.Series([0, 1])
    model = trainer.train_logistic_regression(X, y)

    assert isinstance(model, LogisticRegression)
    assert (tmp_path / "logistic.pkl").exists()


@patch("mlops.models.models.load_config")
def test_train_linear_regression_runs(mock_config, tmp_path):
    mock_config.return_value = {
        "model": {},
        "artifacts": {"preprocessing_pipeline": str(tmp_path / "pipe.pkl")},
        "target": "price"
    }

    trainer = ModelTrainer()
    trainer.model_config = {
        "linear_regression": {
            "params": {"fit_intercept": True},
            "save_path": str(tmp_path / "linear.pkl")
        }
    }

    X = pd.DataFrame([[1, 2], [2, 3], [3, 4]])
    y = pd.Series([10.0, 20.0, 30.0])
    model = trainer.train_linear_regression(X, y)

    assert isinstance(model, LinearRegression)
    assert hasattr(model, "predict")
    assert (tmp_path / "linear.pkl").exists()

    preds = model.predict(X)
    assert np.allclose(preds, [10.0, 20.0, 30.0], atol=1e-1)


@patch("mlops.models.models.load_config")
def test_save_model_creates_file(mock_config, tmp_path):
    mock_config.return_value = {
        "model": {},
        "artifacts": {"preprocessing_pipeline": str(tmp_path / "pipe.pkl")},
        "target": "price"
    }

    trainer = ModelTrainer()
    dummy_model = LinearRegression()
    save_path = tmp_path / "model.pkl"
    trainer._save_model(dummy_model, str(save_path))

    assert save_path.exists()


@patch.object(ModelTrainer, "prepare_data")
@patch.object(ModelTrainer, "train_linear_regression")
@patch.object(ModelTrainer, "train_logistic_regression")
@patch("mlops.models.models.load_config")
def test_train_all_models_success(mock_config, mock_log, mock_lin, mock_prep):
    mock_config.return_value = {
        "model": {
            "linear_regression": {"save_path": "models/tmp_lr.pkl"},
            "logistic_regression": {"save_path": "models/tmp_log.pkl"},
        },
        "target": "price",
        "artifacts": {"preprocessing_pipeline": "models/tmp_pipeline.pkl"}
    }

    mock_prep.return_value = (
        np.array([[1, 2]]), np.array([[1, 2]]),
        pd.Series([10]), pd.Series([1]),
        pd.Series([10]), pd.Series([1])
    )

    trainer = ModelTrainer()
    lin_model = MagicMock()
    log_model = MagicMock()
    mock_lin.return_value = lin_model
    mock_log.return_value = log_model

    result = trainer.train_all_models(pd.DataFrame({"a": [1], "b": [2]}))

    assert result == (lin_model, log_model)
