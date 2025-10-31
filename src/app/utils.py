import os
import jwt
import json
import joblib
import pandas as pd
from datetime import datetime,timezone
from fastapi import HTTPException, Cookie
from src.app.config import SECRET_KEY,ALGORITHM,USERS_FILE,INFERENCE_DATA_PATH
from src.utils.const import scaler_file_name,label_encoders_file_name

# JSON USER DATABASE HELPERS
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def get_current_user(access_token: str = Cookie(None)):
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


def label_churn(pred):
    return "Yes" if int(pred) == 1 else "No"

def log_inference(raw_input: dict, prediction: int,probability: float):
    record = raw_input.copy()
    record['Churn'] = label_churn(prediction)
    record['probability'] = probability
    record['timestamp'] = datetime.now(timezone.utc).isoformat()
    
    df = pd.DataFrame([record])
    os.makedirs(os.path.dirname(INFERENCE_DATA_PATH), exist_ok=True)
    file_exists = os.path.exists(INFERENCE_DATA_PATH)
    df.to_csv(INFERENCE_DATA_PATH, mode='a', header=not file_exists, index=False)

def log_predictions_task(raw_inputs, preds,probs):
    """This function will be executed in the background."""
    for i, pred in enumerate(preds):
        log_inference(raw_input=raw_inputs[i], prediction=float(pred),probability=probs[i])

def read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)

def file_exist(path: str):
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
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

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


