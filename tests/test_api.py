from fastapi.testclient import TestClient
from app.main import app  # Import your FastAPI app
import pytest

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

def test_predict_endpoint():
    """Test the /predict endpoint with sample data."""
    sample_data = {
        "ETHUSDT_price": 1800.0,
        "BNBUSDT_price": 300.0,
        "XRPUSDT_price": 0.5,
        "ADAUSDT_price": 0.3,
        "SOLUSDT_price": 25.0,
        "BTCUSDT_funding_rate": 0.0001,
        "ETHUSDT_funding_rate": 0.0001,
        "BNBUSDT_funding_rate": 0.0001,
        "XRPUSDT_funding_rate": 0.0001,
        "ADAUSDT_funding_rate": 0.0001,
        "SOLUSDT_funding_rate": 0.0001
    }
    
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    
    prediction = response.json()
    assert "predicted_btc_price" in prediction
    assert "predicted_price_direction" in prediction
    assert "prediction_probability" in prediction
    assert isinstance(prediction["predicted_btc_price"], float)
    assert isinstance(prediction["predicted_price_direction"], int)
    assert isinstance(prediction["prediction_probability"], float) 