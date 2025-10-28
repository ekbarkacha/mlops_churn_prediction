import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import yaml
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.data_pipeline.data_preprocessing import data_processing_pipeline
from src.data_pipeline.feature_engineering import feature_engineering_pipeline
from src.modeling.model_utils import split_data, evaluate_model, save_model, log_metrics_to_mlflow
from src.utils.logger import get_logger
from src.utils.config import MLFLOW_TRACKING_URI
from src.utils.const import (MLFLOW_EXPERIMENT_NAME,
                             MODEL_ARTIFACTS_PATH,
                             PROCESSED_DATA_DIR,
                             processed_file_name,
                             feature_file_name)
from src.modeling.nn_model import NN,PyTorchWrapper
import json
from pathlib import Path

logger = get_logger(__name__)

# Load Config
with open("src/modeling/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

def tune_and_log_model(model_cls, model_name, X_train, y_train, X_test, y_test, params, all_metrics, log_model_func):
    """
    This function if for hyperparameter tuning and MLflow logging.
    It also supports GridSearchCV and RandomizedSearchCV.
    """
    with mlflow.start_run(run_name=model_name):
        base_params = {k: v for k, v in params.items() if k != "tuning"}
        model = model_cls(**base_params)
        tuning_cfg = params.get("tuning", {})

        if tuning_cfg.get("enabled", False):
            search_type = tuning_cfg.get("search_type", "grid")
            param_grid = tuning_cfg["param_grid"]
            cv = tuning_cfg.get("cv", 3)
            if search_type == "grid":
                search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring="f1")
            else:
                n_iter = tuning_cfg.get("n_iter", 5)
                search = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=n_iter, n_jobs=-1, scoring="f1")
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            mlflow.log_params(search.best_params_)
            logger.info(f"{model_name} Best Params: {search.best_params_}")
        else:
            model.fit(X_train, y_train)
            best_model = model
            mlflow.log_params(base_params)

        # Evaluate
        preds = best_model.predict(X_test)
        metrics = evaluate_model(y_test, preds,best_model.predict_proba(X_test)[:, 1])
        all_metrics[model_name] = metrics
        log_metrics_to_mlflow(metrics)

        # Save & log model
        # model_path = save_model(best_model, f"{MODEL_ARTIFACTS_PATH}/{model_name.lower()}.joblib")
        # mlflow.log_artifact(model_path)
        log_model_func(best_model)

        logger.info(f"{model_name} Metrics: {metrics}")


def train_all_models(df: pd.DataFrame):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    all_metrics = {}

    logger.info("Loading processed data...")
    X_train, X_test, y_train, y_test = split_data(df,test_size=0.2)
    over = SMOTE(random_state=2)
    X_train, y_train = over.fit_resample(X_train, y_train)

    input_example = X_test.iloc[:2].astype("float64")

    # Random Forest
    tune_and_log_model(
        RandomForestClassifier,
        "RandomForest",
        X_train, y_train, X_test, y_test,
        config["random_forest"],
        all_metrics,
        lambda m: mlflow.sklearn.log_model(sk_model=m, name="model",input_example=input_example)
    )

    # XGBoost
    tune_and_log_model(
        xgb.XGBClassifier,
        "XGBoost",
        X_train, y_train, X_test, y_test,
        config["xgboost"],
        all_metrics,
        lambda m: mlflow.xgboost.log_model(xgb_model=m, name="model",input_example=input_example)
    )

    # Neural Network
    nn_cfg = config["neural_net"]
    tuning_cfg = nn_cfg.get("tuning", {})
    best_metrics = None
    best_params = None

    with mlflow.start_run(run_name="NeuralNet"):
        param_grid = tuning_cfg.get("param_grid", {}) if tuning_cfg.get("enabled", False) else {"hidden_units": [nn_cfg["hidden_units"]], "lr": [nn_cfg["lr"]], "batch_size": [nn_cfg["batch_size"]]}
        
        for hu in param_grid["hidden_units"]:
            for lr in param_grid["lr"]:
                for bs in param_grid["batch_size"]:
                    logger.info(f"Training NN with hidden_units={hu}, lr={lr}, batch_size={bs}")
                    input_dim = X_train.shape[1]
                    model = NN(input_dim, hu)

                    if torch.backends.mps.is_available():
                        device = torch.device("mps")
                    elif torch.cuda.is_available():
                        device = torch.device("cuda")
                    else:
                        device = torch.device("cpu")

                    model = model.to(device)
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    X_train_t = torch.from_numpy(X_train.to_numpy(dtype=np.float32))
                    y_train_t = torch.from_numpy(y_train.to_numpy(dtype=np.float32).reshape(-1, 1))
                    X_test_t = torch.from_numpy(X_test.to_numpy(dtype=np.float32))

                    train_loader = DataLoader(
                        TensorDataset(X_train_t, y_train_t),
                        batch_size=bs, shuffle=True
                    )

                    for epoch in range(nn_cfg["epochs"]):
                        model.train()
                        epoch_loss = 0.0
                        for batch_X, batch_y in train_loader:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                            optimizer.zero_grad(set_to_none=True)
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()

                    # Evaluate
                    model.eval()
                    with torch.no_grad():
                        logits = model(X_test_t.to(device))
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                        preds = (probs > 0.5).astype(int)

                    metrics = evaluate_model(y_test, preds,probs)
                    if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
                        best_metrics = metrics
                        best_params = {"hidden_units": hu, "lr": lr, "batch_size": bs}
                        best_model = model

        mlflow.log_params(best_params)
        log_metrics_to_mlflow(best_metrics)
        all_metrics["neural_net"] = best_metrics
        # torch.save(best_model.state_dict(), f"{MODEL_ARTIFACTS_PATH}/neural_net.pt")
        # mlflow.log_artifact(f"{MODEL_ARTIFACTS_PATH}/neural_net.pt")
        mlflow.pyfunc.log_model(python_model=PyTorchWrapper(best_model.to("cpu")), 
                                 name="model",
                                 input_example=input_example)

    # # Save all metrics
    # Path(MODEL_ARTIFACTS_PATH).mkdir(exist_ok=True)
    # with open(f"{MODEL_ARTIFACTS_PATH}/metrics.json", "w") as f:
    #     json.dump(all_metrics, f, indent=4)

    logger.info(f"Training complete. Check MLflow UI for experiment results. {MLFLOW_TRACKING_URI}")
    print(f"Training complete. Check MLflow UI for experiment results. {MLFLOW_TRACKING_URI}")



def register_best_model():
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        logger.error(f"Experiment {MLFLOW_EXPERIMENT_NAME} not found.")
        return None
    runs = client.search_runs(
        experiment.experiment_id, order_by=["metrics.f1 DESC"], max_results=1
    )

    if not runs:
        logger.warning(f"No runs found for experiment {MLFLOW_EXPERIMENT_NAME}.")
        return None
    best_run = runs[0]
    
    logger.info(f"Best run ID: {best_run.info.run_id}, f1: {best_run.data.metrics['f1']:.4f}")

    model_uri = f"runs:/{best_run.info.run_id}/model"
    registered_model = mlflow.register_model(model_uri, MLFLOW_EXPERIMENT_NAME)
    logger.info(f"Registered model version: {registered_model.version}")

if __name__ == "__main__":
    processed_data_path = data_processing_pipeline()
    df = feature_engineering_pipeline(processed_data_path)
    if not df.empty:
        train_all_models(df)
        register_best_model()
    else:
        logger.error("The resulting DataFrame from feature engineering is empty.")
        raise ValueError("The resulting DataFrame from feature engineering is empty.")
