"""
Module: api/v1/monitoring.py
==============================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Monitoring routes for data drift, metrics reporting, and model performance.
"""
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.util import get_remote_address
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from src.app.core.security import verify_api_key
from src.app.core.deps import require_role
from src.app.models.monitoring import MetricsUpdateRequest, MetricsUpdateResponse
from src.app.services.monitoring_service import MonitoringService
from src.app.ml.preprocessing import InferencePreprocessor
from src.app.config import (
    APP_USERS,
    INFERENCE_DATA_PATH,
    RAW_DATA_PATH,
    EXPECTED_COLUMNS
)
from src.utils.logger import get_logger

logger = get_logger("customer_churn_api")
router = APIRouter(prefix="/monitoring", tags=["Monitoring"])
limiter = Limiter(key_func=get_remote_address)


@router.get("/data_monitoring", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def data_monitoring(
    request: Request,
    format: str = "html",
    window: int = None,
    payload: dict = Depends(require_role(APP_USERS.get(2)))
):
    """
    Generate data quality and drift monitoring reports using Evidently AI.
    Only accessible to Admins.

    Args:
        request: FastAPI request object.
        format: Output format ("html" or "json").
        window: Optional window size for recent samples.
        payload: JWT payload from admin user.

    Returns:
        HTML or JSON report.
    """
    logger.info("Metric report generation requested.")
    monitoring_service = MonitoringService()

    if monitoring_service.file_exists(INFERENCE_DATA_PATH) and monitoring_service.file_exists(RAW_DATA_PATH):
        cur_data = pd.read_csv(INFERENCE_DATA_PATH)
        ref_data = pd.read_csv(RAW_DATA_PATH)

        if window is not None:
            cur_data = cur_data.sort_values(by=["timestamp"], ascending=True)
            n = min(window, len(cur_data))
            cur_data = cur_data.tail(n)

        if cur_data.shape[0] >= 10:
            # Get version folder for preprocessing
            version_folder = request.app.state.model_service.version_folders.get("Production")
            if version_folder:
                preprocessor = InferencePreprocessor(version_dir=version_folder)
                ref_data = preprocessor.data_cleaning(ref_data[EXPECTED_COLUMNS])

            cur_data = cur_data[EXPECTED_COLUMNS]

            report = Report(metrics=[DataDriftPreset()], include_tests=True)
            report.run(reference_data=ref_data, current_data=cur_data)
        else:
            if format.lower() == "html":
                html_content = "<html><body><h3>Not enough production data for metrics evaluation (min 10 data points required).</h3></body></html>"
                return HTMLResponse(content=html_content, status_code=400)
            else:
                return JSONResponse(
                    content={"error": "Not enough production data for metrics evaluation (min 10 data points required)."},
                    status_code=400
                )

    elif monitoring_service.file_exists(RAW_DATA_PATH):
        ref_data = pd.read_csv(RAW_DATA_PATH)
        report = Report(metrics=[DataSummaryPreset()])
        report.run(ref_data)
    else:
        if format.lower() == "html":
            html_content = "<html><body><h3>No data available for metrics.</h3></body></html>"
            return HTMLResponse(content=html_content, status_code=404)
        else:
            return JSONResponse(content={"error": "No data available"}, status_code=404)

    if format.lower() == "html":
        logger.debug("Returning HTML report.")
        return HTMLResponse(content=report.get_html())
    logger.debug("Returning JSON report.")
    return JSONResponse(content=report.as_dict())


@router.post("/update_metrics", response_model=MetricsUpdateResponse, dependencies=[Depends(verify_api_key)])
async def update_metrics(data: dict):
    """
    Centralized Prometheus metrics ingestion endpoint.

    Accepts JSON payloads from different monitoring subsystems:
    - drift: updates data drift metrics
    - pipeline_error: increments pipeline error counters
    - model_decay: updates model performance metrics

    Args:
        data: Metrics update payload.

    Returns:
        Success response with source.
    """
    monitoring_service = MonitoringService()

    # Drift Metrics
    if data.get("from") == "drift":
        model_name = data.get("model_name", "unknown_model")
        drift_share = data.get("drift_share", 0.0)
        drift_count = data.get("drift_count", 0)
        monitoring_service.update_drift_metrics(model_name, drift_share, drift_count)

    # Pipeline Errors
    elif data.get("from") == "pipeline_error":
        pipeline_name = data.get("pipeline_name", "unknown_pipeline")
        monitoring_service.update_pipeline_error(pipeline_name)

    # Model Decay Check
    elif data.get("from") == "model_decay":
        model_name = data.get("model_name", "unknown_model")
        version = str(data.get("version", "0"))
        model_type = data.get("model_type", "unknown_type")
        metrics = data.get("metric", {})
        monitoring_service.update_model_metrics(model_name, version, model_type, metrics)

    return MetricsUpdateResponse(status="ok", source=data.get("from"))


@router.get("/all_version_metrics", dependencies=[Depends(verify_api_key)])
async def all_version_metrics(
    request: Request,
    payload: dict = Depends(require_role(APP_USERS.get(2)))
):
    """
    Get metrics for all model versions from MLflow registry.
    Only accessible to Admins.

    Args:
        request: FastAPI request object.
        payload: JWT payload from admin user.

    Returns:
        Dictionary of version -> metrics.
    """
    model_service = request.app.state.model_service
    return model_service.model_metrics_cache


@router.get("/iframe_data_monitoring_proxy")
async def iframe_proxy(
    api_key: str = Query(...),
    token: str = Query(...),
    format: str = "html",
    window: int = 50
):
    """
    Proxy endpoint for embedding data monitoring in iframe.

    Args:
        api_key: API key for authentication.
        token: JWT token for authorization.
        format: Output format.
        window: Window size for data.

    Returns:
        HTML response with monitoring report.
    """
    from fastapi import FastAPI
    app = FastAPI()

    # Import main app to access routes
    from src.app.main import app as main_app
    client = TestClient(main_app)

    headers = {"x-api-key": api_key}
    cookies = {"access_token": token}

    response = client.get(
        f"/monitoring/data_monitoring?format={format}&window={window}",
        headers=headers,
        cookies=cookies
    )

    return HTMLResponse(content=response.text, status_code=response.status_code)
