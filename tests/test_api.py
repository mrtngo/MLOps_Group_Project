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
