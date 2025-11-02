import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import requests
from airflow import DAG    
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator, ShortCircuitOperator
from airflow.utils.dates import days_ago
from src.utils.const import EXPECTED_COLUMNS,RAW_DATA_DIR,raw_file_name,INFERENCE_DATA_DIR,inference_file_name
import subprocess
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Set project root
project_root = os.getcwd()

default_args = {
    "owner": "mlops_team"
}


# COMBINE ORIGINAL AND PRODUCTION DATA
def combine_data(original_path, current_path, combined_path):
    # Load original data
    original_df = pd.read_csv(original_path)
    original_df = original_df[EXPECTED_COLUMNS]
    
    # Load current (new) data
    current_df = pd.read_csv(current_path)
    current_df = current_df[EXPECTED_COLUMNS]

    # Combine both datasets
    combined_df = pd.concat([original_df, current_df], ignore_index=True)
    
    # Drop duplicate rows
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)
    
    # Save
    combined_df.to_csv(combined_path, index=False)
    
    print(f"Combined data saved to {combined_path}")

# CHECK MODEL DECAY
def check_model_decay_callable(**kwargs):
    print("Running check_model_decay_callable", flush=True)
    try:
        result = subprocess.run(
            ["python", "src/monitoring/check_model_decay.py"],
            capture_output=True,
            text=True,
            check=False
        )
        print("Subprocess executed", flush=True)
        print("STDOUT:", result.stdout, flush=True)
        print("STDERR:", result.stderr, flush=True)
        print(f"Return code: {result.returncode}", flush=True)
        if result.returncode == 1:
            return "dvc_pull"
        return "no_model_decay"
    except subprocess.CalledProcessError as e:
        print("Subprocess failed", flush=True)
        print("Error:", e, flush=True)
        raise

# CHECK FOR DATA DRIFT
def check_data_drift_callable(**kwargs):
    try:
        result = subprocess.run(
            ["python", "src/monitoring/check_data_drift.py"],
            capture_output=True,
            text=True,
            check=False
        )
        print("STDOUT:", result.stdout)
        return "combine_data" if result.returncode == 1 else "no_drift_detected"
    except subprocess.CalledProcessError as e:
        print("Error running drift check script:", e)
        raise


## Airflow Task to Trigger GitHub Workflow
def trigger_github_workflow_training(**context):
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
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

    if response.status_code != 204:
        raise Exception(f"{REPO_OWNER},{REPO_NAME},{WORKFLOW_TRAINING_FILE},Failed to trigger GitHub Actions workflow")




# DAG FOR DATA DRIFT
with DAG(
    dag_id="customer_churn_drift_pipeline",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['data drift','training',"github trigger"],
) as data_drift_dag:
    
    # Check for drift
    check_drift = BranchPythonOperator(
        task_id="check_drift",
        python_callable=check_data_drift_callable,
        provide_context=True,
    )

    # Combine inference data with raw if drift detected
    combine_data_task = PythonOperator(
        task_id="combine_data",
        python_callable=combine_data,
        op_kwargs={
            "original_path": f"{project_root}/{RAW_DATA_DIR}/{raw_file_name}",
            "current_path": f"{project_root}/{INFERENCE_DATA_DIR}/{inference_file_name}",
            "combined_path": f"{project_root}/{RAW_DATA_DIR}/{raw_file_name}",
        },
    )

    # No drift and no decay
    no_drift_detected = BashOperator(
        task_id="no_drift_detected",
        bash_command="echo 'No drift detected, skipping retrain.'",
    )

    # Trigger github workflow training
    trigger_workflow_task = PythonOperator(
        task_id='trigger_github_workflow_training_task',
        python_callable=trigger_github_workflow_training,
        provide_context=True,
    )

    # Workflow
    check_drift >> [combine_data_task, no_drift_detected]
    combine_data_task >> trigger_workflow_task


# DAG FOR MODEL DECAY
with DAG(
    dag_id="customer_churn_model_decay",
    default_args=default_args,
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['model decay','training',"github trigger"],
) as model_decay_dag:
    
    # Check for model decay
    check_model_decay = BranchPythonOperator(
        task_id="check_model_decay",
        python_callable=check_model_decay_callable,
    )

    # Combine inference data with raw if drift detected
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

    # Trigger github workflow training
    trigger_workflow_task = PythonOperator(
        task_id='trigger_github_workflow_training_task',
        python_callable=trigger_github_workflow_training,
        provide_context=True,
    )

    # Workflow
    check_model_decay >> [combine_data_task, no_model_decay]
    combine_data_task >> trigger_workflow_task
