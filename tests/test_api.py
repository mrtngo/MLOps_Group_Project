import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from fastapi.testclient import TestClient

from app.main import app  # Import your FastAPI app

client = TestClient(app)


def test_health_check():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_read_root():
    """Test the root / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_missing_models(monkeypatch):
    """Test /predict returns 503 if models or preprocessor are not loaded."""
    from app import main as app_main

    monkeypatch.setattr(app_main, "reg_model", None)
    monkeypatch.setattr(app_main, "class_model", None)
    monkeypatch.setattr(app_main, "preprocessor_pipeline", None)
    # Provide a valid payload with all required fields
    valid_payload = {
        "ETHUSDT_price": 1.0,
        "BNBUSDT_price": 1.0,
        "XRPUSDT_price": 1.0,
        "ADAUSDT_price": 1.0,
        "SOLUSDT_price": 1.0,
        "BTCUSDT_funding_rate": 0.1,
        "ETHUSDT_funding_rate": 0.1,
        "BNBUSDT_funding_rate": 0.1,
        "XRPUSDT_funding_rate": 0.1,
        "ADAUSDT_funding_rate": 0.1,
        "SOLUSDT_funding_rate": 0.1,
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 503
    assert "Models or preprocessor not loaded" in response.text


def test_predict_invalid_input():
    """Test /predict returns 422 for invalid input data."""
    # Missing required fields
    response = client.post("/predict", json={"ETHUSDT_price": 100})
    assert response.status_code == 422


def test_predict_batch_invalid_file():
    """Test /predict_batch with a non-CSV file returns 400."""
    response = client.post(
        "/predict_batch", files={"file": ("input.txt", b"not a csv", "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.text


def test_predict_batch_missing_models(monkeypatch, tmp_path):
    """Test /predict_batch returns 503 if models or preprocessor are not loaded."""
    from app import main as app_main

    monkeypatch.setattr(app_main, "reg_model", None)
    monkeypatch.setattr(app_main, "class_model", None)
    monkeypatch.setattr(app_main, "preprocessor_pipeline", None)
    # Create a valid CSV
    import pandas as pd

    df = pd.DataFrame([{f: 1.0 for f in app_main.ALL_FEATURES}])
    file_path = tmp_path / "input.csv"
    df.to_csv(file_path, index=False)
    with open(file_path, "rb") as f:
        response = client.post(
            "/predict_batch", files={"file": ("input.csv", f, "text/csv")}
        )
    assert response.status_code == 503
    assert "Models or preprocessor not loaded" in response.text


def make_dummy_preprocessor():
    class DummyScaler:
        def transform(self, X):
            return X.values
    return {
        "scaler": DummyScaler(),
        "selected_features_reg": [
            "ETHUSDT_price",
            "BNBUSDT_price",
            "XRPUSDT_price",
            "ADAUSDT_price",
            "SOLUSDT_price",
            "ETHUSDT_funding_rate",
            "XRPUSDT_funding_rate",
            "ADAUSDT_funding_rate",
            "SOLUSDT_funding_rate",
            "BTCUSDT_funding_rate",
        ],
        "selected_features_class": [
            "BNBUSDT_price",
            "XRPUSDT_price",
            "ADAUSDT_price",
            "ETHUSDT_funding_rate",
            "ADAUSDT_funding_rate",
        ],
        "all_feature_cols": [
            "ETHUSDT_price",
            "BNBUSDT_price",
            "XRPUSDT_price",
            "ADAUSDT_price",
            "SOLUSDT_price",
            "BTCUSDT_funding_rate",
            "ETHUSDT_funding_rate",
            "BNBUSDT_funding_rate",
            "XRPUSDT_funding_rate",
            "ADAUSDT_funding_rate",
            "SOLUSDT_funding_rate",
        ],
    }

class DummyRegModel:
    def predict(self, X):
        return [42.0] * len(X)

class DummyClassModel:
    def predict(self, X):
        return [1] * len(X)
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.1, 0.9]] * len(X))

def valid_payload():
    return {
        "ETHUSDT_price": 1.0,
        "BNBUSDT_price": 1.0,
        "XRPUSDT_price": 1.0,
        "ADAUSDT_price": 1.0,
        "SOLUSDT_price": 1.0,
        "BTCUSDT_funding_rate": 0.1,
        "ETHUSDT_funding_rate": 0.1,
        "BNBUSDT_funding_rate": 0.1,
        "XRPUSDT_funding_rate": 0.1,
        "ADAUSDT_funding_rate": 0.1,
        "SOLUSDT_funding_rate": 0.1,
    }

def test_predict_happy_path(monkeypatch):
    from app import main as app_main
    monkeypatch.setattr(app_main, "reg_model", DummyRegModel())
    monkeypatch.setattr(app_main, "class_model", DummyClassModel())
    monkeypatch.setattr(app_main, "preprocessor_pipeline", make_dummy_preprocessor())
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 200
    data = response.json()
    assert "price_prediction" in data
    assert "direction_prediction" in data
    assert "direction_probability" in data

def test_predict_internal_error(monkeypatch):
    from app import main as app_main
    monkeypatch.setattr(app_main, "reg_model", DummyRegModel())
    monkeypatch.setattr(app_main, "class_model", DummyClassModel())
    # Make preprocessor raise error
    def bad_performer(*a, **kw):
        raise Exception("fail!")
    monkeypatch.setattr(app_main, "preprocessor_pipeline", make_dummy_preprocessor())
    monkeypatch.setattr(app_main, "perform_prediction", bad_performer)
    response = client.post("/predict", json=valid_payload())
    assert response.status_code == 500
    assert "fail!" in response.text

def test_predict_batch_happy_path(monkeypatch, tmp_path):
    from app import main as app_main
    monkeypatch.setattr(app_main, "reg_model", DummyRegModel())
    monkeypatch.setattr(app_main, "class_model", DummyClassModel())
    monkeypatch.setattr(app_main, "preprocessor_pipeline", make_dummy_preprocessor())
    import pandas as pd
    df = pd.DataFrame([valid_payload(), valid_payload()])
    file_path = tmp_path / "input.csv"
    df.to_csv(file_path, index=False)
    with open(file_path, "rb") as f:
        response = client.post(
            "/predict_batch", files={"file": ("input.csv", f, "text/csv")}
        )
    assert response.status_code == 200
    data = response.json()
    assert "price_prediction" in data
    assert len(data["price_prediction"]) == 2
    assert "direction_prediction" in data
    assert len(data["direction_prediction"]) == 2
    assert "direction_probability" in data
    assert len(data["direction_probability"]) == 2

def test_predict_batch_internal_error(monkeypatch, tmp_path):
    from app import main as app_main
    monkeypatch.setattr(app_main, "reg_model", DummyRegModel())
    monkeypatch.setattr(app_main, "class_model", DummyClassModel())
    monkeypatch.setattr(app_main, "preprocessor_pipeline", make_dummy_preprocessor())
    def bad_performer(*a, **kw):
        raise Exception("fail-batch!")
    monkeypatch.setattr(app_main, "perform_prediction", bad_performer)
    import pandas as pd
    df = pd.DataFrame([valid_payload()])
    file_path = tmp_path / "input.csv"
    df.to_csv(file_path, index=False)
    with open(file_path, "rb") as f:
        response = client.post(
            "/predict_batch", files={"file": ("input.csv", f, "text/csv")}
        )
    assert response.status_code == 500
    assert "fail-batch!" in response.text

def test_predict_batch_empty_csv(monkeypatch, tmp_path):
    from app import main as app_main
    monkeypatch.setattr(app_main, "reg_model", DummyRegModel())
    monkeypatch.setattr(app_main, "class_model", DummyClassModel())
    monkeypatch.setattr(app_main, "preprocessor_pipeline", make_dummy_preprocessor())
    file_path = tmp_path / "empty.csv"
    with open(file_path, "w") as f:
        f.write("")
    with open(file_path, "rb") as f:
        response = client.post(
            "/predict_batch", files={"file": ("empty.csv", f, "text/csv")}
        )
    # Should return 500 due to pandas read_csv error
    assert response.status_code == 500
    assert "No columns to parse from file" in response.text or "Error" in response.text

def test_predict_batch_missing_columns(monkeypatch, tmp_path):
    from app import main as app_main
    monkeypatch.setattr(app_main, "reg_model", DummyRegModel())
    monkeypatch.setattr(app_main, "class_model", DummyClassModel())
    monkeypatch.setattr(app_main, "preprocessor_pipeline", make_dummy_preprocessor())
    import pandas as pd
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    file_path = tmp_path / "bad.csv"
    df.to_csv(file_path, index=False)
    with open(file_path, "rb") as f:
        response = client.post(
            "/predict_batch", files={"file": ("bad.csv", f, "text/csv")}
        )
    assert response.status_code == 500
    assert "Error" in response.text or "not in index" in response.text
