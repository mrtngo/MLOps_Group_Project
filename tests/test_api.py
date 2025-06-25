import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
