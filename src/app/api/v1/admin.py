"""
Module: api/v1/admin.py
========================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Admin-only routes for data management, feedback, and model reloading.
"""
import os
import pandas as pd
from io import StringIO
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.app.core.security import verify_api_key, verify_admin_api_key
from src.app.core.deps import require_role
from src.app.models.prediction import UploadDataResponse, CanaryPercentageRequest, CanaryPercentageResponse
from src.app.services.monitoring_service import MonitoringService
from src.app.config import (
    APP_USERS,
    INFERENCE_DATA_PATH,
    RAW_DATA_PATH,
    EXPECTED_COLUMNS
)
from src.data_pipeline.data_preprocessing import validate_schema
from src.utils.logger import get_logger
from src.monitoring.metrics import API_ERRORS

logger = get_logger("customer_churn_api")
router = APIRouter(prefix="/admin", tags=["Admin"])
limiter = Limiter(key_func=get_remote_address)


@router.get("/feedback_data", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def get_feedback_data(
    request: Request,
    limit: int = 50,
    payload: dict = Depends(require_role(APP_USERS.get(2)))
):
    """
    Get recent model inference results for feedback.
    Only accessible to Admins.

    Args:
        request: FastAPI request object.
        limit: Maximum number of records to return (default 50).
        payload: JWT payload from admin user.

    Returns:
        List of recent inference records.
    """
    method = request.method
    endpoint = request.url.path
    try:
        logger.info("Admin requested feedback data retrieval.")
        monitoring_service = MonitoringService()
        df = monitoring_service.get_feedback_data(limit=limit)

        if df.empty:
            raise HTTPException(status_code=404, detail="No inference data found.")

        records = df.to_dict(orient="records")
        logger.debug(f"Returning top {limit} recent inference records.")

        return records

    except HTTPException as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status=e.status_code).inc()
        logger.error(f"Feedback data retrieval failed: {e.detail}")
        raise


@router.post("/feedback", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def feedback(
    request: Request,
    file: UploadFile = File(...),
    payload: dict = Depends(require_role(APP_USERS.get(2)))
):
    """
    Upload human-verified prediction data (feedback).
    Only accessible to Admins.

    Args:
        request: FastAPI request object.
        file: CSV file with feedback data.
        payload: JWT payload from admin user.

    Returns:
        Success message with merge statistics.
    """
    method = request.method
    endpoint = request.url.path
    logger.info("Admin uploading feedback data file.")
    logger.debug(f"Uploaded file: {file.filename}")

    try:
        # Read uploaded CSV
        contents = await file.read()
        await file.close()
        updated_df = pd.read_csv(StringIO(contents.decode("utf-8")))

        if "customerID" not in updated_df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'customerID' column")

        monitoring_service = MonitoringService()
        result = monitoring_service.merge_feedback(updated_df)

        logger.info("Feedback data merged successfully.")
        return result

    except HTTPException as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status=e.status_code).inc()
        logger.error(f"Feedback failed: {e.detail}")
        raise


@router.post("/upload_data", response_model=UploadDataResponse, dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def upload_training_data(
    request: Request,
    file: UploadFile = File(...),
    payload: dict = Depends(require_role(APP_USERS.get(2)))
):
    """
    Upload new or updated training data.
    Only accessible to Admins.

    Args:
        request: FastAPI request object.
        file: CSV file with training data.
        payload: JWT payload from admin user.

    Returns:
        Upload success message with statistics.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    try:
        # Read uploaded CSV
        contents = await file.read()
        await file.close()
        new_df = pd.read_csv(StringIO(contents.decode("utf-8")))

        if new_df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

        # Validate schema
        try:
            validate_schema(new_df, EXPECTED_COLUMNS)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

        # Load existing training data if it exists
        merged_path = RAW_DATA_PATH
        if os.path.exists(merged_path):
            existing_df = pd.read_csv(merged_path)
            combined_df = pd.concat([existing_df, new_df])
            combined_df.drop_duplicates(subset=["customerID"], keep="last", inplace=True)
        else:
            combined_df = new_df

        # Save merged CSV
        combined_df.to_csv(merged_path, index=False)

        return UploadDataResponse(
            message="Training data uploaded and merged successfully.",
            file_saved_at=merged_path,
            rows=len(combined_df),
            columns=len(combined_df.columns)
        )

    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload training data: {str(e)}")


@router.post("/set_canary_percentage", response_model=CanaryPercentageResponse, dependencies=[Depends(verify_admin_api_key)])
async def set_canary_percentage(
    request: Request,
    canary_request: CanaryPercentageRequest
):
    """
    Dynamically update canary traffic percentage (0.0–1.0).
    Only accessible via admin API key (for Airflow/automation).

    Args:
        request: FastAPI request object.
        canary_request: Canary percentage (0.0 to 1.0).

    Returns:
        Success message with updated percentage.
    """
    model_service = request.app.state.model_service
    model_service.set_canary_weight(canary_request.percentage)

    return CanaryPercentageResponse(
        message=f"Canary traffic updated to {canary_request.percentage * 100:.0f}%"
    )


@router.post("/reload_models", dependencies=[Depends(verify_admin_api_key)])
async def reload_models(request: Request):
    """
    Reload models from MLflow registry without restarting the app.
    Only accessible via admin API key (for Airflow/automation).

    Args:
        request: FastAPI request object.

    Returns:
        Success message.
    """
    logger.info("Model reload triggered by Airflow or Admin.")
    model_service = request.app.state.model_service
    await model_service.load_models()

    return {"status": "success", "message": "Models reloaded successfully."}
