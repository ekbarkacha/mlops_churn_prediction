#Kaggle Dataset
KAGGLE_DATASET="blastchar/telco-customer-churn"

# Dir Paths
LOG_DIR = "logs"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

# File Names
raw_file_name = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
processed_file_name = "customers_cleaned.csv"
feature_file_name = "customers_features.csv"

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
