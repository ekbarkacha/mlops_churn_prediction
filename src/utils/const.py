"""
Constants module - Project-wide constants
"""
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Dataset info
KAGGLE_DATASET_NAME = "blastchar/telco-customer-churn"
RAW_FILE_NAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_FILE_NAME = "telco_churn_processed.csv"
FEATURE_FILE_NAME = "telco_churn_features.csv"

# Directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DATA_DIR = PROJECT_ROOT / "data" / "features"
PREPROCESSORS_DIR = PROJECT_ROOT / "artifacts" / "preprocessors"
LOGS_DIR = PROJECT_ROOT / "logs"

# Preprocessor artifact names
LABEL_ENCODERS_FILE = "label_encoders.pkl"
SCALER_FILE = "scaler.pkl"

# Expected columns (21 colonnes du dataset Telco)
EXPECTED_COLUMNS = [
    'customerID',
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges',
    'TotalCharges',
    'Churn'
]

# Target column
TARGET_COLUMN = "Churn"

# Feature types
CATEGORICAL_FEATURES = [
    'gender', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']

# SeniorCitizen is already numeric (0/1)
BINARY_FEATURES = ['SeniorCitizen']

# Columns to exclude from encoding
EXCLUDE_FROM_ENCODING = ['customerID', TARGET_COLUMN]

# Features to drop (based on analysis - low importance/redundancy)
FEATURES_TO_DROP = [
    'customerID',      # Identifiant unique, pas de valeur prédictive
    'PhoneService',    # Redondant avec MultipleLines
    'gender',          # Faible corrélation avec Churn
    'StreamingTV',     # Service optionnel, faible impact
    'StreamingMovies', # Service optionnel, faible impact
    'MultipleLines',   # Corrélé avec autres features télécom
    'InternetService'  # Redondant, capturé par autres features
]

# Features to scale (numerical only, excluding target)
FEATURES_TO_SCALE = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Final features after engineering (14 features including target)
FINAL_FEATURES = [
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges',
    'TotalCharges',
    'Churn'
]

# MLflow Configuration
# Construct MLflow tracking URI (compatible with Windows paths)
mlruns_dir = PROJECT_ROOT / "mlruns"
# Convert Windows backslashes to forward slashes for file:// URI
MLFLOW_TRACKING_URI = "file:///" + str(mlruns_dir).replace("\\", "/")
MLFLOW_EXPERIMENT_NAME = "churn_prediction"

# Model Artifacts
MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "models"

# Model Config
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.yaml"

# Create directories
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
PREPROCESSORS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)