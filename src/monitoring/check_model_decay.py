"""
Module: check_model_decay.py
============================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description:
------------
This module monitors **model decay** — degradation of model performance over time 
when evaluated on new inference data.  
It retrieves the **latest production model** from MLflow, evaluates its performance 
(F1 score and related metrics) on recent inference logs, and determines whether retraining 
should be triggered.

If the model’s F1 score falls below a predefined threshold (`MODEL_THRESHOLD`), 
the module signals that retraining is required.

Workflow Summary:
-----------------
1. Connect to MLflow Tracking Server and fetch the latest Production model.
2. Download (or load cached) model artifacts and preprocessors.
3. Load and clean inference data (including true labels).
4. Generate predictions using the production model.
5. Evaluate model performance using standard classification metrics.
6. Push metrics to FastAPI monitoring service for Prometheus.
7. Return the F1 score for Airflow DAG decision-making.

"""
# Imports and setup
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Custom utility imports
from src.app.model_wrapper import UniversalMLflowWrapper
from src.utils.config import MLFLOW_TRACKING_URI,MODEL_THRESHOLD
from src.utils.const import INFERENCE_DATA_DIR,inference_file_name,MLFLOW_EXPERIMENT_NAME,MODEL_DIR,EXPECTED_COLUMNS
from src.data_pipeline.data_preprocessing import validate_schema
from src.monitoring.metrics import push_metrics_to_fastapi
from src.modeling.model_utils import evaluate_model
from src.utils.logger import get_logger
from src.data_pipeline.data_ingestion import ingest_from_csv

logger = get_logger(__name__)

def check_model_decay():
    """
    Steps:
    ------
    1. Connect to MLflow Tracking Server and fetch the latest Production model.
    2. Download (or load cached) model artifacts and preprocessors.
    3. Load and clean inference data (including true labels).
    4. Generate predictions using the production model.
    5. Evaluate model performance using standard classification metrics.
    6. Push metrics to FastAPI monitoring service for Prometheus.
    7. Return the F1 score for Airflow DAG decision-making.

    Returns
    -------
    float
        F1 score of the current production model on recent inference data.

    """
    # Setting mlflow tracking uri
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Connect to MLflow Tracking Server
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        logger.info(f"Connected to MLflow at {MLFLOW_TRACKING_URI}")

        # Get all registered model versions
        versions = client.search_model_versions(f"name='{MLFLOW_EXPERIMENT_NAME}'")

    except MlflowException as e:
        logger.warning(f"Could not connect to MLflow: {e}")       
        versions = []

    if not versions:
        logger.error(f"No model versions found for {MLFLOW_EXPERIMENT_NAME}. Skipping model load.")
        push_metrics_to_fastapi({"from":"pipeline_error","pipeline_name":"model_decay_check"})
        raise RuntimeError(f"No versions found for '{MLFLOW_EXPERIMENT_NAME}' in MLflow registry.")
    
    # Choose which stage to load
    TARGET_STAGE = "Production"  # "Archived", "Production", "Staging"

    # Filter versions by stage
    stage_versions = [v for v in versions if v.current_stage == TARGET_STAGE]

    if not stage_versions:
        logger.warning(f"No model versions found in stage '{TARGET_STAGE}'. Falling back to latest version.")
        latest_version = sorted(versions, key=lambda v: int(v.version))[-1]
    else:
        # Pick latest version in that stage
        latest_version = sorted(stage_versions, key=lambda v: int(v.version))[-1]

    model_uri = f"models:/{latest_version.name}/{latest_version.version}"
    version_folder = os.path.join(MODEL_DIR, f"v{latest_version.version}")

    # Get run name from the latest version
    run_info = client.get_run(latest_version.run_id)
    run_name = run_info.data.tags.get("mlflow.runName", "Unnamed Run")

    # Versioned model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download and cache model locally (if not cached)
    if not os.path.exists(version_folder):
        logger.info(f"Downloading model v{latest_version.version} from MLflow registry...")
        
        # Load model from MLflow
        model = mlflow.pyfunc.load_model(model_uri)

        # Save local copy for caching
        mlflow.pyfunc.save_model(
            path=version_folder,
            python_model=model._model_impl.python_model
        )

        # Download preprocessing artifacts (eg scaler, encoders)
        logger.info(f"Downloading preprocessor artifacts for run_id: {latest_version.run_id}")
        client.download_artifacts(run_id=latest_version.run_id, path="preprocessors", dst_path=version_folder)
        
        logger.info(f"Model version {latest_version.version} cached locally at {version_folder}.")
    else:
        logger.info(f"Loading cached model version {latest_version.version} from {version_folder}")
        model = mlflow.pyfunc.load_model(version_folder)


    # Wrap model inside UniversalMLflowWrapper
    model = UniversalMLflowWrapper(model._model_impl.python_model.model,
                                                version_dir=version_folder,
                                                background_data=None)

    logger.info(f"Model {MLFLOW_EXPERIMENT_NAME} version {latest_version.version} loaded successfully.")


    # Load and preprocess inference data
    INFERENCE_DATA_PATH = os.path.join(INFERENCE_DATA_DIR, inference_file_name)
    inference_df = ingest_from_csv(INFERENCE_DATA_PATH)

    # Validate schema
    validate_schema(inference_df, EXPECTED_COLUMNS[:-1])

    inference_df = inference_df[EXPECTED_COLUMNS]

    # Encode target variable 'Churn'
    if "Churn" in inference_df.columns:
        inference_df["Churn"] = inference_df["Churn"].map({'No': 0, 'Yes': 1})

    X = inference_df.drop(columns=["Churn"])

    y_true = inference_df["Churn"]

    # Predict and evaluate model performance
    y_pred, y_pred_prob = model.predict(model_input=X, return_proba=True, both=True)

    metrics_data = evaluate_model(y_true, y_pred, y_pred_prob)
    f1 = metrics_data.get("f1",0)

    logger.info(f"F1 score on recent inference data: {f1:.4f}")
    # Push metrics to FastAPI monitoring endpoint for prometheus
    push_metrics_to_fastapi({"from":"model_decay",
                             "model_name":MLFLOW_EXPERIMENT_NAME,
                             "version":latest_version.version,
                             "model_type":run_name, "metric": metrics_data})

    return f1

if __name__ == "__main__":
    """
    Entry point for manual execution.
    Evaluates model decay and exits with status code:
    - 0 : Model performance acceptable
    - 1 : Model decay detected (trigger retraining)
    """
    f1 = check_model_decay()
    if f1 < MODEL_THRESHOLD:
        print("Model performance degraded. Trigger retrain.",flush=True)
        logger.info("Model performance degraded. Trigger retrain.")
        sys.exit(1)
    else:
        print("Model performance is acceptable.",flush=True)
        logger.info("Model performance is acceptable.")
        sys.exit(0)
