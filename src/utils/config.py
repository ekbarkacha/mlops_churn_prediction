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
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Drift Threshold
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD",0.5))

# Model decay baseline
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD",0.80))

# Fastapi 
FASTAPI_URL = os.getenv("FASTAPI_URL")

# Admin api key
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")