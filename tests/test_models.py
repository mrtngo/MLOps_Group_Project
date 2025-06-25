import sys
import os
import pickle
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from mlops.models.models import (
    ModelTrainer,
    get_training_and_testing_data,
    train_model,
)


# === Your original tests ===

@patch("mlops.models.models.load_config")
@patch("mlops.models.models.smote_oversample")
@patch("mlops.models.models.select_features")
@patch("mlops.models.models.scale_features")
@patch("mlops.models.models.split_data")
@patch("mlops.models.models.prepare_features")
@patch("mlops.models.models.create_price_direction_label")
@patch("mlops.models.models.define_features_and_label")
def test_prepare_data(
    mock_define,
    mock_create_label,
    mock_prepare_feat,
    mock_split,
    mock_scale,
    mock_select,
    mock_smote,
    mock_config,
):
    mock_config.return_value = {
        "model": {},
        "target": "price",
        "artifacts": {"preprocessing_pipeline": "models/test_pipeline.pkl"},
    }

    mock_define.return_value = (["f1", "f2"], "price")
    mock_create_label.return_value = pd.DataFrame(
        {"f1": [1], "f2": [2], "price": [10], "price_direction": [1]}
    )
    mock_prepare_feat.return_value = (
        np.array([[1, 2]]),
        pd.Series([10]),
        pd.Series([1]),
    )
    mock_split.return_value = (
        np.array([[1, 2]]),
        np.array([[3, 4]]),
        pd.Series([10]),
        pd.Series([20]),
    )

    scaler = StandardScaler().fit([[1, 2], [3, 4]])
    mock_scale.return_value = (np.array([[1.1, 2.2]]), np.array([[1.3, 2.5]]), scaler)

    mock_select.side_effect = lambda df, f, target=None: f
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
        "target": "price",
    }

    trainer = ModelTrainer()
    trainer.model_config = {
        "logistic_regression": {
            "params": {},
            "save_path": str(tmp_path / "logistic.pkl"),
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
        "target": "price",
    }

    trainer = ModelTrainer()
    trainer.model_config = {
        "linear_regression": {
            "params": {"fit_intercept": True},
            "save_path": str(tmp_path / "linear.pkl"),
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
        "target": "price",
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
        "artifacts": {"preprocessing_pipeline": "models/tmp_pipeline.pkl"},
    }

    mock_prep.return_value = (
        np.array([[1, 2]]),
        np.array([[1, 2]]),
        pd.Series([10]),
        pd.Series([1]),
        pd.Series([10]),
        pd.Series([1]),
    )

    trainer = ModelTrainer()
    lin_model = MagicMock()
    log_model = MagicMock()
    mock_lin.return_value = lin_model
    mock_log.return_value = log_model

    result = trainer.train_all_models(pd.DataFrame({"a": [1], "b": [2]}))

    assert result == (lin_model, log_model)


def test_train_model_none():
    with pytest.raises(ValueError):
        train_model(None)


def test_save_model_invalid_path():
    trainer = ModelTrainer()

    class DummyModel:
        pass

    invalid_path = "/invalid_dir/invalid_model.pkl"
    with pytest.raises(Exception):
        trainer._save_model(DummyModel(), invalid_path)


# === Extended tests for 100% coverage ===

@pytest.fixture(autouse=True)
def patch_full_config(tmp_path, monkeypatch):
    fake_conf = {
        "model": {
            "linear_regression": {
                "params": {"fit_intercept": False},
                "save_path": str(tmp_path / "lr.pkl"),
            },
            "logistic_regression": {
                "params": {},
                "save_path": str(tmp_path / "log.pkl"),
            },
        },
        "artifacts": {
            "preprocessing_pipeline": str(tmp_path / "pipe_dir/pipe.pkl")
        },
        "target": "price",
    }
    monkeypatch.setattr(
        "mlops.models.models.load_config",
        lambda *args, **kwargs: fake_conf,
    )
    return tmp_path


def test_ensure_output_directories_creates(tmp_path):
    trainer = ModelTrainer()
    assert os.path.isdir("models")
    assert os.path.isdir(os.path.dirname(trainer.config["artifacts"]["preprocessing_pipeline"]))


def test_save_preprocessing_pipeline_writes_correct_content(patch_full_config):
    trainer = ModelTrainer()
    trainer.scaler = StandardScaler().fit([[0, 1], [2, 3]])
    trainer.selected_features_reg = ["f1", "f2"]
    trainer.selected_features_class = ["f2"]
    trainer.feature_cols = ["f1", "f2"]
    trainer._save_preprocessing_pipeline()

    path = trainer.config["artifacts"]["preprocessing_pipeline"]
    assert os.path.exists(path)
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    assert set(pipeline.keys()) == {
        "scaler",
        "selected_features_reg",
        "selected_features_class",
        "all_feature_cols",
    }


def test_prepare_data_saves_pipeline_and_returns_shapes(tmp_path, monkeypatch):
    import mlops.models.models as mmod
    monkeypatch.setattr(mmod, "define_features_and_label", lambda: (["a", "b"], "price"))
    monkeypatch.setattr(mmod, "create_price_direction_label", lambda df, lbl: df.assign(price_direction=df["price"] > 0))
    monkeypatch.setattr(mmod, "prepare_features", lambda df, feats, lbl: (df[feats].values, df["price"], df["price_direction"]))
    monkeypatch.setattr(mmod, "split_data", lambda X, y: (X[:1], X[1:], y[:1], y[1:]))
    scaler = StandardScaler().fit([[0, 1], [1, 0]])
    monkeypatch.setattr(mmod, "scale_features", lambda df, feats: (scaler.transform(df), scaler.transform(df), scaler))
    monkeypatch.setattr(mmod, "select_features", lambda df, feats, target=None: feats)
    monkeypatch.setattr(mmod, "smote_oversample", lambda X, y: (X, y))

    trainer = ModelTrainer()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "price": [10, -5]})
    X_tr, X_cl, y_tr, y_cl_bal, y_te_reg, y_te_cl = trainer.prepare_data(df)
    assert X_tr.shape[1] == 2
    assert X_cl.shape[1] == 2
    assert len(y_tr) == 1
    assert os.path.exists(trainer.config["artifacts"]["preprocessing_pipeline"])


def test_train_model_success(monkeypatch):
    monkeypatch.setattr(
        "mlops.models.models.ModelTrainer.train_all_models",
        lambda self, df: ("lr", "log"),
    )
    res = train_model(pd.DataFrame({"x": [1]}))
    assert res == ("lr", "log")


def test_train_model_none_raises():
    with pytest.raises(ValueError):
        train_model(None)
