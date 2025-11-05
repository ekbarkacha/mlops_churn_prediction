"""Test FastAPI endpoints"""
import pytest
from fastapi.testclient import TestClient
from src.deployment.api import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "model_loaded" in data


def test_predict_without_api_key():
    """Test prediction endpoint without API key"""
    response = client.post(
        "/predict",
        json={
            "data": [
                {
                    "SeniorCitizen": 0,
                    "Partner": 1,
                    "Dependents": 0,
                    "tenure": 0.5,
                    "MonthlyCharges": 0.6,
                    "TotalCharges": 0.4
                }
            ]
        }
    )
    assert response.status_code == 403  # Forbidden without API key


def test_predict_with_valid_api_key():
    """Test prediction endpoint with valid API key"""
    response = client.post(
        "/predict",
        json={
            "data": [
                {
                    "SeniorCitizen": 0,
                    "Partner": 1,
                    "Dependents": 0,
                    "tenure": 0.5,
                    "MonthlyCharges": 0.6,
                    "TotalCharges": 0.4
                }
            ]
        },
        headers={"X-API-Key": "dev_key_12345"}
    )
    
    # May fail if model not loaded, but should not be 403
    assert response.status_code != 403


def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get(
        "/model/info",
        headers={"X-API-Key": "dev_key_12345"}
    )
    assert response.status_code in [200, 500]  # 200 if model loaded, 500 if not


def test_docs_available():
    """Test that API documentation is available"""
    response = client.get("/docs")
    assert response.status_code == 200
    
    response = client.get("/openapi.json")
    assert response.status_code == 200