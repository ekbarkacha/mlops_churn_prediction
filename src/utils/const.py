"""
Module: const.py 
=======================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description: 
------------
This module defines **static constants** used across the entire MLOps project.
It helps maintain consistency in dataset paths, filenames, MLflow experiment names, 
and expected schema definitions etc.  

By centralizing constants, we:
- Avoid hardcoded strings across multiple scripts.
- Ensure that all modules (Airflow DAGs, FastAPI, training, monitoring) 
  use the same configuration.
- Simplify updates to dataset or directory structures.
"""
#Kaggle Dataset
KAGGLE_DATASET="blastchar/telco-customer-churn"

# Dir Paths
LOG_DIR = "data/logs"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
USERS_DATA_DIR = "data/users"
INFERENCE_DATA_DIR = "data/inference"
MODEL_DIR = "data/models"
PREPROCESSORS = "data/preprocessors"
REPORTS_DIR = "data/reports"


# File Names
raw_file_name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
processed_file_name = "customers_cleaned.csv"
feature_file_name = "customers_features.csv"
users_file_name = "users.json"
inference_file_name = "production_data.csv"
label_encoders_file_name = "label_encoder.joblib"
scaler_file_name = "scaler.joblib"
data_drift_report_json = "data_drift_report.json"
data_drift_report_html = "data_drift_report.html"

# Expected Colums
EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

# Experiment name in MLflow
MLFLOW_EXPERIMENT_NAME = "Customer_Churn_Prediction"
MODEL_ARTIFACTS_PATH = "artifacts"
