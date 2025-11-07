"""
DAG: Canary Deployment Pipeline
===============================

Author: AIMS-AMMI STUDENT
Created: October/November 2025
Description:
------------
This Airflow DAG orchestrates a canary deployment workflow for machine learning models.
It performs incremental rollout of a newly deployed canary model, gradually increasing 
its traffic exposure while monitoring performance. If the canary model performs well at 
each stage, it is automatically promoted to production.

Workflow Summary:
-----------------
1. Check if an active canary model exists in MLflow.
2. If found, begin a progressive rollout (5% → 25% → 100%) with waiting periods.
3. After each rollout step, validate canary performance using model prediction logs.
4. If performance is acceptable, proceed to the next rollout step.
5. If the canary passes all steps, automatically promote it to "Production" in MLflow.

Components:
-----------
- **BranchPythonOperator**: Determines if a canary model exists.
- **PythonOperator**: Performs rollout and promotion actions.
- **TimeDeltaSensor**: Waits between rollout increments.
- **ShortCircuitOperator**: Conditionally stops DAG if canary fails evaluation.
- **BashOperator**: Outputs messages for logging purposes.

"""

# Import required dependencies
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.operators.python import ShortCircuitOperator
import mlflow
from datetime import datetime, timedelta
import requests
from dotenv import load_dotenv

# Custom utility imports
from src.utils.const import MLFLOW_EXPERIMENT_NAME
from src.utils.config import MLFLOW_TRACKING_URI, ADMIN_API_KEY,FASTAPI_URL
from src.monitoring.check_canary import check_canary_from_logs
from src.modeling.model_utils import set_model_stage
from src.utils.logger import get_logger

# Initialization
load_dotenv()
logger = get_logger(__name__)

# Sett canary percentage url
SET_CANARY_ENDPOINT = f"{FASTAPI_URL}/set_canary_percentage"

# Canary rollout configuration
CANARY_STEPS = [0.05, 0.25, 1.0]
WAIT_BETWEEN_STEPS = timedelta(hours=6)

# Default Airflow arguments
default_args = {"owner": "mlops_team"}

# Check if canary exists
def check_canary(**context):
    """Check MLflow for an existing 'Staging' (canary) model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    models = client.search_model_versions(f"name='{MLFLOW_EXPERIMENT_NAME}'")

    # Looking for a model currently in 'Staging'
    for m in models:
        if m.current_stage == "Staging":
            logger.info(f"Active canary detected: version {m.version}")
            context['ti'].xcom_push(key="canary_info", value={
                "model_name": m.name,
                "model_version": m.version,
                "stage": m.current_stage
            })
            return "canary_exists"

    logger.info("No active canary found. DAG will stop here.")
    return "no_canary"


# Set canary rollout weight
def set_canary_weight(weight: float):
    """Adjust canary traffic exposure via FastAPI endpoint."""
    logger.info(f"Setting canary traffic to {weight*100:.0f}%")
    headers = {"x-api-key": ADMIN_API_KEY}
    response = requests.post(SET_CANARY_ENDPOINT, params={"percentage": weight}, headers=headers, timeout=10)
    response.raise_for_status()
    logger.info(f"Canary rollout {weight*100:.0f}% applied successfully.")
    return weight


# Evaluate canary performance
def canary_passed(weight: float, **context):
    """Evaluate canary health based on prediction logs."""
    result = check_canary_from_logs()
    # Continue DAG only if mentics suggests promotion
    return bool(result.get("promote")) 

# Promote canary to production
def promote_canary(**context):
    """Promote the canary model to 'Production' if evaluation passed."""
    canary_info = context['ti'].xcom_pull(key="canary_info", task_ids="check_canary")
    if not canary_info:
        logger.warning("No canary info found. Skipping promotion.")
        return

    model_name = canary_info["model_name"]
    model_version = canary_info["model_version"]
    stage = canary_info["stage"]

    logger.info("Promoting canary model to Production...")
    set_model_stage(
        logger,
        MLFLOW_TRACKING_URI,
        model_name,
        model_version,
        "Production",
        archive_existing_versions=True,
        reload=True,
    )
    logger.info(f"Canary model version {model_version} promoted to Production successfully!")


# CANARY DEPLOYMENT DAG
with DAG(
    dag_id="canary_deployment_pipeline",
    default_args=default_args,
    description="Incremental canary rollout with evaluation and auto-promotion",
    schedule_interval="0 2 * * *",## 02:00 am,
    start_date=datetime(2025, 11, 4),
    catchup=False,
    tags=["canary", "deployment"],
) as dag:

    # Determine if canary exists
    check_canary_task = BranchPythonOperator(
        task_id="check_canary",
        python_callable=check_canary,
        provide_context=True,
    )

    # Canary exists -> continue rollout
    canary_exists = BashOperator(
        task_id="canary_exists",
        bash_command="echo 'Active canary found. Proceeding with rollout...'",
    )

    # No canary -> stop DAG
    no_canary = BashOperator(
        task_id="no_canary",
        bash_command="echo 'No canary found. Exiting DAG.'",
    )

    # Branching logic
    check_canary_task >> [canary_exists, no_canary]

    tasks_map = {}
    previous_evaluate = None

    # Canary rollout loop
    for i, weight in enumerate(CANARY_STEPS):
        # Apply rollout percentage
        rollout = PythonOperator(
            task_id=f"canary_rollout_{int(weight*100)}",
            python_callable=set_canary_weight,
            op_kwargs={"weight": weight},
        )
        tasks_map[rollout.task_id] = rollout

        # Wait before evaluating
        wait = TimeDeltaSensor(
            task_id=f"wait_after_{int(weight*100)}",
            delta=WAIT_BETWEEN_STEPS,
        )
        tasks_map[wait.task_id] = wait

        # Evaluate canary metric
        evaluate = ShortCircuitOperator(
            task_id=f"evaluate_canary_{int(weight*100)}",
            python_callable=canary_passed,
            op_kwargs={"weight": weight},
            provide_context=True,
        )

        tasks_map[evaluate.task_id] = evaluate

        rollout >> wait >> evaluate

        # Chain from previous evaluation if exists
        if previous_evaluate:
            previous_evaluate >> rollout
        else:
            canary_exists >> rollout

        # Define next task in sequence
        next_task_id = (
            f"canary_rollout_{int(CANARY_STEPS[i+1]*100)}"
            if i < len(CANARY_STEPS) - 1
            else "promote_canary_to_production"
        )

        if next_task_id in tasks_map:
            evaluate >> tasks_map[next_task_id]

        previous_evaluate = evaluate

    # Promote to Production    
    promote_task = PythonOperator(
        task_id="promote_canary_to_production",
        python_callable=promote_canary,
        provide_context=True,
    )
    tasks_map[promote_task.task_id] = promote_task

    # Final evaluation to promotion
    previous_evaluate >> promote_task