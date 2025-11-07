"""
Module: utils.py
==========================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description: 
------------
Utility functions and classes for the FastAPI churn prediction application.

Includes:
- User management helpers (JSON-based)
- JWT authentication and token decoding
- Model selection logic (staging/production/canary)
- Inference logging
- Safe CSV reading and file existence checks
- Preprocessing for inference data

Classes:
- InferenceDatapreprocer: Preprocesses raw input data for inference, applying the same transformations as during training.
"""
# Imports and setup
import os
import jwt
import json
import joblib
import random 
import pandas as pd
from datetime import datetime,timezone
from fastapi import HTTPException, Cookie,Request

# Custom utility imports
from src.app.config import SECRET_KEY,ALGORITHM,USERS_FILE,INFERENCE_DATA_PATH
from src.utils.const import scaler_file_name,label_encoders_file_name,EXPECTED_COLUMNS

# JSON USER DATABASE HELPERS
def load_users() -> dict:
    """
    Load users from a JSON file.

    Returns:
        dict: Dictionary of users.
    """
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users: dict) -> None:
    """
    Save users to a JSON file.

    Args:
        users (dict): Dictionary of users to save.
    """
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def get_current_user(access_token: str = Cookie(None)):
    """
    Decode JWT token and return the current user.

    Args:
        access_token (str): JWT access token from cookie.

    Returns:
        dict: User information.

    Raises:
        HTTPException: If token is missing or invalid.
    """
    if not access_token:
        raise HTTPException(status_code=401, detail="Missing token")
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        users = load_users()
        if username is None or username not in users:
            raise HTTPException(status_code=401, detail="Invalid token")
        return users[username]
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    
def get_model(request: Request):
    """
    Select the model to use for inference based on canary deployment logic.

    Args:
        request (Request): FastAPI request object containing app state.

    Returns:
        tuple: (model, stage, version)
    """
    models = request.app.state.models
    versions = request.app.state.version_folder
    canary_weight = request.app.state.canary_weight

    if not models:
        return None, None, None

    rand_val = random.random()
    if "Staging" in models and rand_val < canary_weight:
        stage = "Staging"
    else:
        stage = "Production" if "Production" in models else next(iter(models.keys()))

    model = models[stage]
    version = versions[stage].split("/")[-1]

    return model, stage, version


def label_churn(pred):
    """
    Convert numeric prediction to 'Yes'/'No' label.

    Args:
        pred (int): Numeric prediction (0 or 1).

    Returns:
        str: 'Yes' for 1, 'No' for 0.
    """
    return "Yes" if int(pred) == 1 else "No"

def log_inference(raw_input: dict, prediction: int,probability: float,stage: str,version: str,latency: float):
    """
    Log a single inference to CSV.

    Args:
        raw_input (dict): Raw input features.
        prediction (int): Predicted class (0/1).
        probability (float): Probability of positive class.
        stage (str): Model stage (Production/Staging).
        version (str): Model version.
        latency (float): Time taken for prediction in seconds.
    """
    record = raw_input.copy()
    record['Churn'] = label_churn(prediction)
    record['probability'] = probability
    record['timestamp'] = datetime.now(timezone.utc).isoformat()
    record['model_stage'] = stage
    record['model_version'] = version
    record["latency"] = latency

    COLUMNS = EXPECTED_COLUMNS+['probability','timestamp','model_stage','model_version',"latency"]

    df = pd.DataFrame([record])
    df = df.reindex(columns=COLUMNS)
    os.makedirs(os.path.dirname(INFERENCE_DATA_PATH), exist_ok=True)
    file_exists = os.path.exists(INFERENCE_DATA_PATH)
    df.to_csv(INFERENCE_DATA_PATH, mode='a', header=not file_exists, index=False)

def log_predictions_task(raw_inputs, preds,probs,stage,version,latency):
    """
    Log batch predictions in background task.

    Args:
        raw_inputs (list): List of raw input dictionaries.
        preds (list): List of predictions.
        probs (list): List of probabilities.
        stage (str): Model stage.
        version (str): Model version.
        latency (float): Prediction latency.
    """
    for i, pred in enumerate(preds):
        log_inference(raw_input=raw_inputs[i], 
                      prediction=float(pred),
                      probability=probs[i],
                      stage=stage,
                      version=version,
                      latency=latency)

def read_csv_safe(path: str) -> pd.DataFrame:
    """
    Safely read a CSV file. Returns empty dataframe if file doesn't exist or is empty.

    Args:
        path (str): Path to CSV file.

    Returns:
        pd.DataFrame: Dataframe read from CSV or empty dataframe.
    """
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)

def file_exist(path: str):
    """
    Check if file exists.

    Args:
        path (str): File path.

    Returns:
        bool: True if file exists, False otherwise.
    """
    if os.path.exists(path):
        return True
    return False

# Inference Data preprocessing
class InferenceDatapreprocer():
    def __init__(self,version_dir=None):
        self.preprocessors_path = os.path.join(version_dir, "preprocessors")
        self.scaler = joblib.load(os.path.join(self.preprocessors_path, scaler_file_name))
        self.encoders = joblib.load(os.path.join(self.preprocessors_path, label_encoders_file_name))

    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy(deep=True)
        # Cleaning
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        # Convert TotalCharges to numeric, coercing invalid values to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Fill NaN values with 0
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

        # Ensure 'SeniorCitizen' is treated as categorical
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')

        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns

        # Fill missing values for numeric columns with median
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)

        # Fill missing values for categorical columns with mode
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply same preprocessing as in training:
        - data cleaning
        - encoding categorical
        - feature selection
        - scaling
        """

        ######## Preprocessing #######
        df = df.copy(deep=True)
        # Cleaning
        # Convert TotalCharges to numeric, coercing invalid values to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Fill NaN values with 0
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

        # Ensure 'SeniorCitizen' is treated as categorical
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')

        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns

        # Fill missing values for numeric columns with median
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)

        # Fill missing values for categorical columns with mode
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)

        # Encode categorical columns
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)

        # Feature_creation

        # Feature Transformation

        # Feature selection
        drop_features = ['customerID','PhoneService', 'gender', 'StreamingTV', 'StreamingMovies', 'MultipleLines', 'InternetService']
        df.drop(columns=drop_features, inplace=True, errors='ignore')

        # Feature Scaling (numeric columns)
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df[num_cols] = self.scaler.transform(df[num_cols])

        return df


