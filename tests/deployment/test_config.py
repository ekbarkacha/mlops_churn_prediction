"""Test deployment configuration"""
import pytest
from src.deployment.config import config


def test_config_loads():
    """Test that config loads without errors"""
    assert config.APP_NAME == "Customer Churn Prediction API"
    assert config.PORT == 8000
    assert isinstance(config.API_KEYS, list)
    assert len(config.API_KEYS) > 0


def test_rate_limiting_config():
    """Test rate limiting configuration"""
    assert config.RATE_LIMIT_PER_MINUTE > 0
    assert config.RATE_LIMIT_PER_HOUR > 0
    assert config.RATE_LIMIT_PER_HOUR > config.RATE_LIMIT_PER_MINUTE


def test_model_config():
    """Test model configuration"""
    # MLFLOW_TRACKING_URI peut être 'file:///mlruns' ou 'http://...'
    assert config.MLFLOW_TRACKING_URI is not None
    assert len(config.MLFLOW_TRACKING_URI) > 0
    
    # Vérifier que c'est un URI valide (commence par file:// ou http://)
    assert (
        config.MLFLOW_TRACKING_URI.startswith("file://") or
        config.MLFLOW_TRACKING_URI.startswith("http://") or
        config.MLFLOW_TRACKING_URI.startswith("https://")
    )
    
    # Experiment name correct
    assert config.MLFLOW_EXPERIMENT_NAME == "churn_prediction"
    assert len(config.MLFLOW_EXPERIMENT_NAME) > 0


def test_batch_config():
    """Test batch processing configuration"""
    assert config.MAX_BATCH_SIZE > 0
    assert config.BATCH_CHUNK_SIZE > 0
    assert config.MAX_BATCH_SIZE >= config.BATCH_CHUNK_SIZE


def test_security_config():
    """Test security configuration"""
    assert isinstance(config.API_KEY_ENABLED, bool)
    assert isinstance(config.RATE_LIMIT_ENABLED, bool)
    
    if config.API_KEY_ENABLED:
        assert len(config.API_KEYS) > 0


def test_cors_config():
    """Test CORS configuration"""
    assert isinstance(config.CORS_ENABLED, bool)
    assert isinstance(config.CORS_ORIGINS, list)
    
    # Check origins format
    for origin in config.CORS_ORIGINS:
        assert origin.startswith("http://") or origin.startswith("https://")