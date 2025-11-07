"""
Module: model_utils.py 
=======================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description: 
------------
Utility functions for model management, evaluation and lifecycle control.

This module provides:
  - Data splitting helpers for training and testing
  - Model evaluation metric computation
  - MLflow integration for logging metrics and managing model stages
  - Serialization utilities (joblib)
  - Integration with FastAPI to trigger model reloads
"""
# Imports
import os
import requests
import joblib
import mlflow
import mlflow.pyfunc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

# Fastapi 
if os.getenv("RUNNING_IN_DOCKER"):
    FASTAPI_URL = os.getenv("FASTAPI_URL")
else:
    FASTAPI_URL = os.getenv("FASTAPI_URL_LOCAL")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")


def split_data(df, target_col="Churn",test_size=0.2):
    """
    Splits a given DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): Input dataset.
        target_col (str): Target column to predict (default: "Churn").
        test_size (float): Fraction of data to allocate to the test set.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)
    return  X_train, X_test, y_train, y_test

# Model Evaluation
def evaluate_model(y_true, y_pred, y_pred_prob):
    """
    Compute key classification metrics including F1, AUC, accuracy, etc.

    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        y_pred_prob (array-like): Predicted probabilities.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_prob),
        "f1": f1_score(y_true, y_pred)
    }
    return metrics

def save_model(model, path):
    """
    Save the model to disk using Joblib serialization.

    Args:
        model: Trained model object.
        path (str): Destination path to save the model.

    Returns:
        str: Path to the saved model file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path

def log_metrics_to_mlflow(metrics: dict):
    """
    Log model metrics to the currently active MLflow run.

    Args:
        metrics (dict): Dictionary of metric name-value pairs.
    """
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

class WrappedModel(mlflow.pyfunc.PythonModel):
    """
    MLflow-compatible wrapper for generic ML models.

    This enables model saving/loading using the MLflow PyFunc format.
    """
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        """
        Perform inference on input data.

        Args:
            context: MLflow context (unused).
            model_input: Input data (e.g., pandas DataFrame).

        Returns:
            np.ndarray: Predicted labels.
        """
        return self.model.predict(model_input)

def set_model_stage(logger,MLFLOW_TRACKING_URI: str,MLFLOW_EXPERIMENT_NAME: str,version:str,stage: str,archive_existing_versions=False,reload=False):
    """
    Update the stage of a model version in MLflow registry and optionally
    trigger a reload on the FastAPI inference server.

    Args:
        logger: Logger object for tracking operations.
        MLFLOW_TRACKING_URI (str): MLflow tracking server URI.
        MLFLOW_EXPERIMENT_NAME (str): Registered model name.
        version (str): Model version number.
        stage (str): Target stage ("Archived", "Production", or "Staging").
        archive_existing_versions (bool): Archive older versions if True.
        reload (bool): Trigger FastAPI model reload if True.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    try:
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.transition_model_version_stage(
            name=MLFLOW_EXPERIMENT_NAME,
            version=version,
            stage=stage, #"Archived", "Production", "Staging"
            archive_existing_versions=archive_existing_versions
        )
        logger.info(f"New stage for model version: {version} and taken to {stage} stage.")
        if reload:
            reload_model_to_fastapi(logger)
    except mlflow.exceptions.MlflowException as e:
        logger.warning(f"Could not set model version {version} new stage to {stage}: {e}")
    


def reload_model_to_fastapi(logger):
    """
    Sends a request to FastAPI service to reload models in memory.

    Args:
        logger: Logger instance for event tracking.
    """
    url = f"{FASTAPI_URL}/reload_models"
    headers = {"x-api-key": ADMIN_API_KEY}
    
    try:
        response = requests.post(url=url, headers=headers, timeout=10)
        if response.status_code == 200:
            logger.info("Model reload triggered successfully on FastAPI server.")
        else:
            logger.warning(f"Failed to reload model. Status code: {response.status_code}, Response: {response.text}")
    except requests.RequestException as e:
        logger.error(f"Error while triggering FastAPI reload: {e}")


