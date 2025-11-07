"""
Module: config.py
=======================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description: 
------------
This module centralizes configuration management for the MLOps pipeline.  
It securely loads environment variables from a `.env` file and makes them 
available to all pipeline components (data ingestion, training, monitoring, etc.).

This ensures consistency across:
- Airflow DAGs
- FastAPI inference service
- MLflow tracking integration
- Monitoring and alerting components

Features:
---------
- Automatically loads `.env` variables using `python-dotenv`   
- Provides centralized access to key environment variables

"""
# Imports
import os
from dotenv import load_dotenv

load_dotenv()

#Setup kaggle
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Setup Mlflow uri (note its aslo in src/app/config)
if os.getenv("RUNNING_IN_DOCKER"):
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
else:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_LOCAL")

# Drift Threshold
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD",0.5))

# Model decay baseline
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD",0.80))

# Fastapi 
if os.getenv("RUNNING_IN_DOCKER"):
    FASTAPI_URL = os.getenv("FASTAPI_URL")
else:
    FASTAPI_URL = os.getenv("FASTAPI_URL_LOCAL")


# Admin api key
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")