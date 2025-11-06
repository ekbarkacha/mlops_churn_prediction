"""
DAGs: Customer Churn Monitoring Pipelines
=========================================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description:
------------
This module defines two Airflow DAGs that monitor data quality/drift and model performance
for a customer churn prediction system. The DAGs automatically detect data drift or
model decay and trigger retraining via a GitHub Actions workflow if thresholds are breached.

There are two main workflows:

1. **Data Drift Pipeline (`customer_churn_drift_pipeline`)**
   - Detects statistical changes in production data using 'Evidently AI.
   - If drift is detected, merges new and historical data.
   - Checks for active canary deployments.
   - Triggers retraining via GitHub Actions.

2. **Model Decay Pipeline (`customer_churn_model_decay`)**
   - Checks if the model’s predictive performance (e.g., F1-score) has degraded below a threshold.
   - If so, merges inference data with the original dataset.
   - If the score is within 10% or 5% above the threshold, sends a warning email indicating potential model decay.
   - Handles canary models if any exist.
   - Initiates retraining workflow.

Common Tasks:
-------------
- **Canary Handling**: Detects and promotes/archives canary models in MLflow.
- **Data Merging**: Combines historical and inference data for retraining.
- **GitHub Actions Trigger**: Automates model retraining when triggered.
"""

# Imports and setup
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import requests
import mlflow
import pandas as pd
from airflow import DAG    
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.email import EmailOperator
from airflow.utils.email import send_email
from airflow.utils.trigger_rule import TriggerRule
from airflow.utils.dates import days_ago
from dotenv import load_dotenv

# Custom utility imports
from src.utils.const import EXPECTED_COLUMNS,RAW_DATA_DIR,raw_file_name,INFERENCE_DATA_DIR,inference_file_name,MLFLOW_EXPERIMENT_NAME
from src.utils.config import MODEL_THRESHOLD, MLFLOW_TRACKING_URI
from src.utils.logger import get_logger

# Initialization
load_dotenv()
logger = get_logger(__name__)
project_root = os.getcwd()

# Default Airflow arguments
default_args = {
    "owner": "mlops_team"
}


# COMBINE ORIGINAL AND PRODUCTION DATA
def combine_data(original_path, current_path, combined_path):
    """
    Combine original training data with current (inference) data.

    Parameters
    ----------
    original_path : str
        Path to the original raw dataset.
    current_path : str
        Path to the new production/inference data.
    combined_path : str
        Output path for the merged dataset.
    """
    # Load and align column structure
    original_df = pd.read_csv(original_path)[EXPECTED_COLUMNS]
    current_df = pd.read_csv(current_path)[EXPECTED_COLUMNS]

    # Concatenate and remove duplicates
    combined_df = pd.concat([original_df, current_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    combined_df.to_csv(combined_path, index=False)
    logger.info(f"Combined data saved to {combined_path}")

# CHECK MODEL DECAY
def check_model_decay_callable(**kwargs):
    """
    Check for model performance decay based on monitoring metrics.

    Returns
    -------
    str : next task ID ("combine_data" or "no_model_decay")
    """
    from src.monitoring.check_model_decay import check_model_decay
    f1 = check_model_decay()
    # Push to XCom for next tasks
    kwargs['ti'].xcom_push(key='f1_score', value=f1)
    if f1 < MODEL_THRESHOLD:
        logger.info("Model performance degraded. Retraining will be trigger.")
        kwargs['ti'].xcom_push(key='warning_level', value='retrain')
        return "combine_data"  # triggers retrain
    elif f1 < min(MODEL_THRESHOLD * 1.05,95):
        logger.warning("Model performance is within 5% of retraining threshold — critical warning.")
        kwargs['ti'].xcom_push(key='warning_level', value='critical')
        return "send_warning_email"
    elif f1 < min(MODEL_THRESHOLD * 1.10, 95):
        logger.info("Model performance is within 10% of retraining threshold — early warning.")
        kwargs['ti'].xcom_push(key='warning_level', value='early')
        return "send_warning_email"
    else:
        logger.info("Model performance is healthy.")
        kwargs['ti'].xcom_push(key='warning_level', value='healthy')
        return "no_model_decay"

# Send Email
def send_dynamic_warning_email(**kwargs):
    """Send a dynamic model decay warning email."""
    ti = kwargs['ti']
    warning_level = ti.xcom_pull(task_ids='check_model_decay', key='warning_level')
    f1_score = ti.xcom_pull(task_ids='check_model_decay', key='f1_score')
    model_name = os.getenv("MLFLOW_EXPERIMENT_NAME")
    alert_email = os.getenv("ALERT_EMAIL", "mlops-team@example.com")

    if not warning_level or warning_level == 'healthy':
        print("No warning to send.")
        return

    # Customize subject and body
    if warning_level == "critical":
        subject = f"CRITICAL: {model_name} performance nearing retraining threshold"
        message = f"""
        <h3>Critical Model Performance Warning</h3>
        <p><b>{model_name}</b> F1 score is <b>{f1_score:.2f}</b>, within 5% of the retraining threshold ({MODEL_THRESHOLD}).</p>
        <p>Immediate review is recommended.</p>
        """
    else:  # early warning
        subject = f"Early Warning: {model_name} performance trending down"
        message = f"""
        <h3>Early Model Performance Warning</h3>
        <p><b>{model_name}</b> F1 score is <b>{f1_score:.2f}</b>, within 10% of the retraining threshold ({MODEL_THRESHOLD}).</p>
        <p>Monitor closely to avoid model decay.</p>
        """

    send_email(
        to=alert_email,
        subject=subject,
        html_content=message
    )
    logger.info(f"Sent {warning_level} warning email to {alert_email}")

# CHECK IF PRODUCTION/INFERENCE DATA
def check_production_data_callable(**kwargs):
    """
    Verify if production/inference data file exists.

    Parameters
    ----------
    file_path : str
        Path to the inference data file.
    next_event : str
        Next task ID if file exists.

    Returns
    -------
    str : next task ("next_event" or "no_production_data")
    """
    file_path = kwargs.get("file_path")
    next_event = kwargs.get("next_event")
    if os.path.exists(file_path):
        return next_event 
    else:
        return "no_production_data"

# CHECK FOR DATA DRIFT
def check_data_drift_callable(**kwargs):
    """
    Check for data drift using statistical comparison.

    Returns
    -------
    str : next task ("combine_data" or "no_drift_detected")
    """
    from src.monitoring.check_data_drift import check_drift
    exit_code = check_drift()
    if exit_code==1:
        return "combine_data"
    else:
        return "no_drift_detected"   


def check_active_canary():
    """
    Check for an active canary model in MLflow and handle promotion/archiving.

    Returns
    -------
    str : next task ("handled_canary" or "no_canary")
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    models = client.search_model_versions(f"name='{MLFLOW_EXPERIMENT_NAME}'")

    for m in models:
        if m.current_stage == "Staging":
            logger.info(f"Active canary detected: version {m.version}")

            # Evaluate canary performance
            from src.monitoring.check_canary import check_canary_from_logs
            from src.modeling.model_utils import set_model_stage

            result = check_canary_from_logs()

            if result.get("promote"):
                stage = "Production"
                archive_existing_versions=True
                logger.info("Promoting canary to Production...")  
            else:
                stage = "Archived"
                archive_existing_versions=False
                logger.info("Archiving underperforming canary...")
                
            set_model_stage(logger,
                            MLFLOW_TRACKING_URI,
                            m.name,
                            m.version,
                            stage,archive_existing_versions=archive_existing_versions,
                            reload=True)

            return "handled_canary"

    logger.info("No active canary found.")
    return "no_canary"

## Trigger GitHub Workflow For Training
def trigger_github_workflow_training(**context):
    """
    Trigger GitHub Actions workflow for model retraining.

    Raises
    ------
    Exception
        If GitHub API returns a non-success status.
    """
    load_dotenv()
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
    REPO_OWNER = os.getenv("REPO_OWNER")
    REPO_NAME = os.getenv("REPO_NAME")
    WORKFLOW_TRAINING_FILE = os.getenv("WORKFLOW_TRAINING_FILE")

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/actions/workflows/{WORKFLOW_TRAINING_FILE}/dispatches"

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28"
    }

    payload = {
        "ref": "main"
    }

    response = requests.post(url, headers=headers, json=payload)
    logger.info(f"GitHub API Response: {response.status_code} - {response.text}")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

    if response.status_code != 204:
        raise Exception(f"Failed to trigger GitHub workflow: {REPO_OWNER}/{REPO_NAME}/{WORKFLOW_TRAINING_FILE}")

# DAG FOR DATA DRIFT
with DAG(
    dag_id="customer_churn_drift_pipeline",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['data drift','training',"github trigger"],
) as data_drift_dag:
    
    # Check if production/inference data is available
    check_production_data = BranchPythonOperator(
        task_id="check_production_data",
        python_callable=check_production_data_callable,
        op_kwargs={
            "file_path": f"{project_root}/{INFERENCE_DATA_DIR}/{inference_file_name}",
            "next_event": "check_drift"
        },
        provide_context=True
    )

    # If no production data found, end pipeline
    no_production_data = BashOperator(
        task_id="no_production_data",
        bash_command="echo 'Production data not found.'",
    )

    # Check for data drift
    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_data_drift_callable,
        provide_context=True,
    )
    
    # Merge data if drift detected
    combine_data_task = PythonOperator(
        task_id="combine_data",
        python_callable=combine_data,
        op_kwargs={
            "original_path": f"{project_root}/{RAW_DATA_DIR}/{raw_file_name}",
            "current_path": f"{project_root}/{INFERENCE_DATA_DIR}/{inference_file_name}",
            "combined_path": f"{project_root}/{RAW_DATA_DIR}/{raw_file_name}",
        },
    )

    # If no drift, log and stop
    no_drift_detected = BashOperator(
        task_id="no_drift_detected",
        bash_command="echo 'No drift detected, skipping retrain.'",
    )

    # Check for active canary model
    check_canary_task = BranchPythonOperator(
        task_id="check_for_canary",
        python_callable=check_active_canary,
    )

    handled_canary = BashOperator(
        task_id="handled_canary",
        bash_command="echo 'Active canary handled, continuing...'",
    )
    no_canary = BashOperator(
        task_id="no_canary",
        bash_command="echo 'No active canary, continuing...'",
    )

    # Merge point after canary handling
    merge_after_canary = BashOperator(
        task_id="merge_after_canary",
        bash_command="echo 'Proceeding after canary check'",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )
    
    # Trigger retraining via GitHub
    trigger_workflow_task = PythonOperator(
        task_id='trigger_github_workflow_training_task',
        python_callable=trigger_github_workflow_training,
        provide_context=True,
    )

    # DAG flow
    check_production_data >> [check_drift, no_production_data]
    check_drift >> [combine_data_task, no_drift_detected]
    combine_data_task >> check_canary_task 
    check_canary_task >> [handled_canary, no_canary]
    handled_canary >> merge_after_canary
    no_canary >> merge_after_canary
    merge_after_canary >> trigger_workflow_task


# DAG FOR MODEL DECAY
with DAG(
    dag_id="customer_churn_model_decay",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['model decay','training',"github trigger"],
) as model_decay_dag:
    # Check if production data exists
    check_production_data = BranchPythonOperator(
        task_id="check_production_data",
        python_callable=check_production_data_callable,
        op_kwargs={
            "file_path": f"{project_root}/{INFERENCE_DATA_DIR}/{inference_file_name}",
            "next_event": "check_model_decay"
        },
        provide_context=True
    )
    # Send email alert when performance start to reduce
    send_warning_email = PythonOperator(
    task_id="send_warning_email",
    python_callable=send_dynamic_warning_email,
    provide_context=True,
    )

    # No production data
    no_production_data = BashOperator(
        task_id="no_production_data",
        bash_command="echo 'Production data not found.'",
    )

    # Evaluate model performance decay
    check_model_decay = BranchPythonOperator(
        task_id="check_model_decay",
        python_callable=check_model_decay_callable,
    )

    # Combine data for retraining if decay detected
    combine_data_task = PythonOperator(
        task_id="combine_data",
        python_callable=combine_data,
        op_kwargs={
            "original_path": f"{project_root}/{RAW_DATA_DIR}/{raw_file_name}",
            "current_path": f"{project_root}/{INFERENCE_DATA_DIR}/{inference_file_name}",
            "combined_path": f"{project_root}/{RAW_DATA_DIR}/{raw_file_name}",
        },
    )

    # No decay
    no_model_decay = BashOperator(
        task_id="no_model_decay",
        bash_command="echo 'No model decay detected, skipping retrain.'",
    )

    # Check and handle canary model
    check_canary_task = BranchPythonOperator(
        task_id="check_for_canary",
        python_callable=check_active_canary,
    )

    handled_canary = BashOperator(
        task_id="handled_canary",
        bash_command="echo 'Active canary handled, continuing...'",
    )
    no_canary = BashOperator(
        task_id="no_canary",
        bash_command="echo 'No active canary, continuing...'",
    )

    merge_after_canary = BashOperator(
        task_id="merge_after_canary",
        bash_command="echo 'Proceeding after canary check'",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )
    
    # Trigger retraining workflow
    trigger_workflow_task = PythonOperator(
        task_id='trigger_github_workflow_training_task',
        python_callable=trigger_github_workflow_training,
        provide_context=True,
    )

    # DAG flow
    check_production_data >> [check_model_decay, no_production_data]
    check_model_decay >> [send_warning_email,combine_data_task, no_model_decay]
    combine_data_task >> check_canary_task 
    check_canary_task >> [handled_canary, no_canary]
    handled_canary >> merge_after_canary
    no_canary >> merge_after_canary
    merge_after_canary >> trigger_workflow_task