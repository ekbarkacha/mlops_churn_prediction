import os
import joblib
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split


def split_data(df, target_col="Churn",test_size=0.2):
    """Split dataset into X and y."""
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)
    return  X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, y_pred_prob):
    """Compute evaluation metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred_prob),
        "f1": f1_score(y_true, y_pred)
    }
    return metrics

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    return path

def log_metrics_to_mlflow(metrics: dict):
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
