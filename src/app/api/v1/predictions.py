"""
Module: api/v1/predictions.py
===============================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Prediction routes for model inference and explainability.
"""
import time
import pandas as pd
from io import StringIO
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.app.core.security import verify_api_key
from src.app.core.deps import get_current_user, require_role
from src.app.models.prediction import AdminPredictionResponse, AgentPredictionResponse
from src.app.services.prediction_service import PredictionService
from src.app.config import APP_USERS, EXPECTED_COLUMNS
from src.utils.logger import get_logger
from src.monitoring.metrics import API_ERRORS

logger = get_logger("customer_churn_api")
router = APIRouter(prefix="/predict", tags=["Predictions"])
limiter = Limiter(key_func=get_remote_address)


def get_model_from_app(request: Request):
    """Helper to get model service from app state."""
    return request.app.state.model_service


@router.post("", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def predict(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Generate churn predictions using MLflow model.

    Admin users can:
    - Upload CSV or send JSON data
    - Receive predictions, probabilities, and SHAP explainability

    Agent users must:
    - Send single JSON input only
    - Receive single prediction and probability

    Args:
        request: FastAPI request object.
        background_tasks: Background task manager for logging.
        file: Optional CSV file upload (admin only).
        current_user: Current authenticated user.

    Returns:
        Predictions with role-specific format.
    """
    method = request.method
    endpoint = request.url.path
    start_time = time.perf_counter()

    # Get model service and select model
    model_service = get_model_from_app(request)
    model, stage, version = model_service.get_model_for_inference()

    if not model:
        logger.warning("Prediction requested but no model is loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    prediction_service = PredictionService()

    try:
        logger.info(f"Prediction request received from user: {current_user['username']} ({current_user['role']})")

        # ADMIN ROLE
        if current_user["role"] == APP_USERS.get(2):
            df = None

            # CASE 1: File was uploaded
            if file is not None:
                logger.debug("CSV file uploaded for prediction.")
                contents = await file.read()
                await file.close()
                df = pd.read_csv(StringIO(contents.decode("utf-8")))

            # CASE 2: JSON body
            else:
                try:
                    data = await request.json()
                    if data:
                        logger.debug("JSON input received for prediction.")
                        df = pd.DataFrame([data])
                except Exception:
                    raise HTTPException(status_code=400, detail="Provide either a CSV file or JSON data.")

            if df is None:
                raise HTTPException(status_code=400, detail="Provide either file or JSON data.")

            # Validate schema
            try:
                prediction_service.validate_input(df)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

            # Keep raw input for logging
            raw_inputs = df.copy(deep=True).to_dict(orient="records")

            # Make predictions with SHAP
            preds, probs, shap_values = prediction_service.make_predictions(
                model=model,
                df=df,
                include_shap=True
            )

            results = df[EXPECTED_COLUMNS[:-1]].copy()
            results["prediction"] = preds
            results["probability"] = probs

            # Log predictions in background
            latency = (time.perf_counter() - start_time) / len(preds)
            background_tasks.add_task(
                prediction_service.log_predictions_batch,
                raw_inputs, preds, probs, stage, version, latency
            )

            # Record metrics
            duration = time.perf_counter() - start_time
            prediction_service.record_metrics(stage, version, duration)

            logger.info(f"Predictions generated for {len(df)} samples.")
            return JSONResponse({
                "role": APP_USERS.get(2),
                "results": results.to_dict(orient="records"),
                "shap_values": shap_values,
            })

        # AGENT ROLE
        elif current_user["role"] == APP_USERS.get(1):
            try:
                data = await request.json()
                logger.debug("JSON input received for prediction.")
            except Exception:
                raise HTTPException(status_code=400, detail="Agents must send JSON data only.")

            if not data:
                raise HTTPException(status_code=400, detail="Agents must send JSON data only.")

            df = pd.DataFrame([data])

            # Validate schema
            try:
                prediction_service.validate_input(df)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

            # Keep raw input for logging
            raw_inputs = df.copy(deep=True).to_dict(orient="records")

            # Make predictions (no SHAP for agents)
            preds, probs, _ = prediction_service.make_predictions(
                model=model,
                df=df,
                include_shap=False
            )

            # Log predictions in background
            latency = (time.perf_counter() - start_time) / len(preds)
            background_tasks.add_task(
                prediction_service.log_predictions_batch,
                raw_inputs, preds, probs, stage, version, latency
            )

            # Record metrics
            duration = time.perf_counter() - start_time
            prediction_service.record_metrics(stage, version, duration)

            logger.info(f"Prediction generated for 1 sample.")

            return JSONResponse({
                "role": APP_USERS.get(1),
                "prediction": int(preds[0]),
                "probability": float(probs[0]),
            })

        else:
            raise HTTPException(status_code=403, detail="Unauthorized role.")

    except HTTPException as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status=e.status_code).inc()
        prediction_service.record_metrics(stage, version, 0, error=True)
        logger.error(f"Prediction error: {e.detail}")
        raise
    except Exception as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status=500).inc()
        prediction_service.record_metrics(stage, version, 0, error=True)
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/explainability", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def model_explainability(
    request: Request,
    file: Optional[UploadFile] = File(None),
    payload: dict = Depends(require_role(APP_USERS.get(2)))
):
    """
    Generate SHAP-based model explainability.
    Only accessible to Admins.

    Args:
        request: FastAPI request object.
        file: Optional CSV file upload.
        payload: JWT payload from admin user.

    Returns:
        SHAP values and feature importance.
    """
    model_service = get_model_from_app(request)
    model, stage, version = model_service.get_model_for_inference()

    if not model:
        logger.warning("Model explainability requested but no model is loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")

    logger.info("Explainability request received.")
    prediction_service = PredictionService()

    df = None

    # CASE 1: File uploaded
    if file is not None:
        contents = await file.read()
        await file.close()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # CASE 2: JSON body
    else:
        try:
            data = await request.json()
            if data:
                df = pd.DataFrame([data])
        except Exception:
            raise HTTPException(status_code=400, detail="Provide either a CSV file or JSON data.")

    if df is None:
        raise HTTPException(status_code=400, detail="Provide either file or JSON data.")

    # Validate schema
    try:
        prediction_service.validate_input(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")

    logger.debug(f"Input data shape: {df.shape}")

    # Generate SHAP values
    df = df[EXPECTED_COLUMNS[:-1]]
    shap_values = model.explain_json(df)

    logger.info("SHAP explanation generated successfully.")
    return JSONResponse({
        "role": APP_USERS.get(2),
        "results": df.to_dict(orient="records"),
        "shap_values": shap_values,
    })
