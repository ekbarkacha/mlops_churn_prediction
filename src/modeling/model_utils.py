"""
Model Utilities - Helper functions for model training and evaluation
"""
import os
import sys
from pathlib import Path
import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.const import TARGET_COLUMN

logger = get_logger(__name__)


def split_data(
    df: pd.DataFrame, 
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 2
) -> tuple:
    """
    Split dataset into train and test sets with stratification.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of target column
        test_size (float): Proportion of test set (0.0 to 1.0)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info(f"🔪 Splitting data (test_size={test_size}, random_state={random_state})...")
    
    try:
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)
        
        logger.info(f"   • Features shape: {X.shape}")
        logger.info(f"   • Target shape: {y.shape}")
        logger.info(f"   • Target distribution: {y.value_counts().to_dict()}")
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y  # Preserve class distribution
        )
        
        logger.info(f"✅ Data split completed:")
        logger.info(f"   • Train set: {X_train.shape[0]} samples")
        logger.info(f"   • Test set: {X_test.shape[0]} samples")
        logger.info(f"   • Train target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"   • Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error(f"❌ Error during data split: {e}")
        raise


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_pred_prob: np.ndarray) -> dict:
    """
    Compute evaluation metrics for classification model.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_pred_prob (np.ndarray): Predicted probabilities
    
    Returns:
        dict: Dictionary of metrics
    """
    try:
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_pred_prob),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }
        
        return metrics
    
    except Exception as e:
        logger.error(f"❌ Error during model evaluation: {e}")
        raise


def save_model(model, path: Path):
    """
    Save model to disk using joblib.
    
    Args:
        model: Model to save
        path (Path): Path to save model
    
    Returns:
        Path: Path where model was saved
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"💾 Model saved to: {path}")
        return path
    
    except Exception as e:
        logger.error(f"❌ Error saving model: {e}")
        raise


def log_metrics_to_mlflow(metrics: dict):
    """
    Log metrics to MLflow.
    
    Args:
        metrics (dict): Dictionary of metrics to log
    """
    try:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        logger.info(f"📊 Logged {len(metrics)} metrics to MLflow")
    
    except Exception as e:
        logger.error(f"❌ Error logging metrics to MLflow: {e}")
        raise


class WrappedModel(mlflow.pyfunc.PythonModel):
    """
    Wrapper for sklearn/XGBoost models to make them MLflow-compatible.
    """
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        """
        Predict using the wrapped model.
        
        Args:
            context: MLflow context (unused)
            model_input: Input data (pandas DataFrame or numpy array)
        
        Returns:
            Predictions
        """
        return self.model.predict(model_input)