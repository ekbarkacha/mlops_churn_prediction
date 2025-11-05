"""
Deployment Configuration - Production Settings
Compatible: Python 3.11.6 + Pydantic 2.5.3
"""
from pydantic_settings import BaseSettings
from typing import List


class DeploymentConfig(BaseSettings):
    """
    Configuration for API deployment
    Reads from environment variables and .env file
    """
    
    # ========== API SETTINGS ==========
    APP_NAME: str = "Customer Churn Prediction API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "Production ML API for predicting customer churn"
    
    # ========== SERVER SETTINGS ==========
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False
    WORKERS: int = 1
    
    # ========== AUTHENTICATION ==========
    API_KEY_ENABLED: bool = True
    API_KEYS: List[str] = [
        "dev_key_12345",
        "prod_key_67890"
    ]
    
    # ========== RATE LIMITING ==========
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # ========== CORS SETTINGS ==========
    CORS_ENABLED: bool = True
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://127.0.0.1:8000"
    ]
    
    # ========== MODEL SETTINGS ==========

    MLFLOW_TRACKING_URI: str = "file:///C:/Users/pc/Desktop/mlops_churn_prediction/mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "churn_prediction"
    MODEL_VERSION: str = "latest"
    
    # ========== BATCH PROCESSING ==========
    MAX_BATCH_SIZE: int = 10000
    BATCH_CHUNK_SIZE: int = 1000
    
    # ========== LOGGING ==========
    LOG_LEVEL: str = "INFO"
    
    # ========== VALIDATION ==========
    MAX_PAYLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    REQUEST_TIMEOUT: int = 30
    
    # Pydantic 2.5.3 Configuration
    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # CRITICAL: Ignore extra variables from .env


# Singleton instance
config = DeploymentConfig()


# Export for convenience
__all__ = ["config", "DeploymentConfig"]