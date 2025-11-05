"""
FastAPI Application - Production-Grade REST API
Customer Churn Prediction API
Compatible: Python 3.11.6 + FastAPI 0.109.0 + Pydantic 2.5.3
"""
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import numpy as np
import pandas as pd

# Local imports
from .config import config
from .auth import verify_api_key, limiter
from .utils import (
    model_loader,
    validate_input_data,
    format_prediction_response,
    get_batch_stats
)

# ========== LOGGING CONFIGURATION ==========
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== PYDANTIC MODELS ==========

class CustomerData(BaseModel):
    """Schema for single customer data"""
    
    # Required fields (from your feature engineering)
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen (0 or 1)")
    Partner: int = Field(..., ge=0, le=1, description="Has partner (0 or 1)")
    Dependents: int = Field(..., ge=0, le=1, description="Has dependents (0 or 1)")
    tenure: float = Field(..., ge=0, le=1, description="Tenure scaled [0-1]")
    MonthlyCharges: float = Field(..., ge=0, le=1, description="Monthly charges scaled [0-1]")
    TotalCharges: float = Field(..., ge=0, le=1, description="Total charges scaled [0-1]")
    
    # Optional fields
    OnlineSecurity: Optional[int] = Field(default=0, ge=0, le=2)
    OnlineBackup: Optional[int] = Field(default=0, ge=0, le=2)
    DeviceProtection: Optional[int] = Field(default=0, ge=0, le=2)
    TechSupport: Optional[int] = Field(default=0, ge=0, le=2)
    Contract: Optional[int] = Field(default=0, ge=0, le=2)
    PaperlessBilling: Optional[int] = Field(default=0, ge=0, le=1)
    PaymentMethod: Optional[int] = Field(default=0, ge=0, le=3)
    
    class Config:
        """Pydantic 2.5.3 config"""
        json_schema_extra = {
            "example": {
                "SeniorCitizen": 0,
                "Partner": 1,
                "Dependents": 0,
                "tenure": 0.5,
                "MonthlyCharges": 0.6,
                "TotalCharges": 0.4,
                "OnlineSecurity": 1,
                "OnlineBackup": 1,
                "DeviceProtection": 0,
                "TechSupport": 1,
                "Contract": 1,
                "PaperlessBilling": 1,
                "PaymentMethod": 0
            }
        }


class PredictionRequest(BaseModel):
    """Schema for batch prediction request"""
    data: List[CustomerData] = Field(..., min_items=1, max_items=1000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "data": [
                    {
                        "SeniorCitizen": 0,
                        "Partner": 1,
                        "Dependents": 0,
                        "tenure": 0.5,
                        "MonthlyCharges": 0.6,
                        "TotalCharges": 0.4
                    }
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    metadata: Dict[str, Any]
    
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "prediction": "Churn",
                        "churn_probability": 0.75,
                        "confidence": 0.75,
                        "risk_level": "HIGH"
                    }
                ],
                "model_info": {
                    "run_id": "abc123",
                    "f1_score": 0.64,
                    "roc_auc": 0.85
                },
                "metadata": {
                    "timestamp": "2024-01-01T12:00:00",
                    "total_predictions": 1
                }
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check"""
    status: str
    timestamp: str
    model_loaded: bool
    model_info: Optional[Dict[str, Any]] = None


    class Config:
        protected_namespaces = ()


# ========== FASTAPI APPLICATION ==========

app = FastAPI(
    title=config.APP_NAME,
    version=config.APP_VERSION,
    description=config.APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add rate limiter to app state
app.state.limiter = limiter


# ========== MIDDLEWARE ==========

# CORS
if config.CORS_ENABLED:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Trusted Hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]
)


# ========== EXCEPTION HANDLERS ==========

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ========== STARTUP/SHUTDOWN EVENTS ==========

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=" * 80)
    logger.info(f"🚀 Starting {config.APP_NAME} v{config.APP_VERSION}")
    logger.info(f"📍 Server: {config.HOST}:{config.PORT}")
    logger.info(f"🔐 Authentication: {'Enabled' if config.API_KEY_ENABLED else 'Disabled'}")
    logger.info(f"⚡ Rate Limiting: {'Enabled' if config.RATE_LIMIT_ENABLED else 'Disabled'}")
    logger.info("=" * 80)
    
    try:
        model_loader.load_model()
        logger.info("✅ Model loaded successfully on startup")
    except Exception as e:
        logger.warning(f"⚠️  Model not loaded on startup: {e}")
        logger.info("   Model will be loaded on first prediction request")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down API server...")


# ========== ENDPOINTS ==========

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": config.APP_NAME,
        "version": config.APP_VERSION,
        "description": config.APP_DESCRIPTION,
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint
    No authentication required
    """
    try:
        model_loaded = model_loader.is_loaded()
        model_info = model_loader.get_model_info() if model_loaded else None
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=model_loaded,
            model_info=model_info
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=False,
            model_info=None
        )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Predict customer churn"
)
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
async def predict(
    request: Request,
    payload: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict customer churn probability
    
    **Authentication:** Required (X-API-Key header)
    
    **Rate Limit:** 60 requests/minute, 1000 requests/hour
    
    **Request Body:**
    - data: List of customer records (max 1000)
    
    **Returns:**
    - predictions: Churn predictions with probabilities
    - model_info: Model metadata
    - metadata: Request metadata
    """
    try:
        logger.info(f"📥 Prediction request from API key: {api_key[:8]}***")
        
        # Convert to DataFrame
        data_dicts = [customer.dict() for customer in payload.data]
        df = validate_input_data(data_dicts)
        
        logger.info(f"✅ Input validated: {len(df)} samples")
        
        # Load model
        model = model_loader.load_model()
        
        # Predict
        predictions_raw = model.predict(df)
        
        # Handle different output formats
        if isinstance(predictions_raw, pd.DataFrame):
            probabilities = predictions_raw.iloc[:, 0].values
        elif isinstance(predictions_raw, np.ndarray):
            probabilities = predictions_raw.flatten()
        else:
            probabilities = np.array(predictions_raw).flatten()
        
        # Binary predictions
        predictions = (probabilities > 0.5).astype(int)
        
        # Format response
        prediction_results = format_prediction_response(
            predictions=predictions,
            probabilities=probabilities,
            input_size=len(df)
        )
        
        # Model info
        model_info = model_loader.get_model_info()
        
        # Stats
        stats = get_batch_stats(predictions)
        
        logger.info(f"✅ Predictions completed: {len(predictions)} samples")
        
        return PredictionResponse(
            predictions=prediction_results,
            model_info=model_info,
            metadata={
                "timestamp": datetime.utcnow().isoformat(),
                "total_predictions": len(predictions),
                "churn_predicted": int(stats['churn_predicted']),
                "churn_rate": float(stats['churn_rate'])
            }
        )
        
    except ValueError as e:
        logger.warning(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.get("/model/info", tags=["Model"])
async def get_model_info(api_key: str = Depends(verify_api_key)):
    """
    Get model information
    
    **Authentication:** Required
    """
    try:
        model_loader.load_model()
        info = model_loader.get_model_info()
        
        return {
            "model_info": info,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model info")


@app.get("/usage", tags=["Monitoring"])
async def get_usage(api_key: str = Depends(verify_api_key)):
    """
    Get API usage info
    
    **Authentication:** Required
    """
    return {
        "api_key": f"{api_key[:8]}***",
        "rate_limits": {
            "per_minute": config.RATE_LIMIT_PER_MINUTE,
            "per_hour": config.RATE_LIMIT_PER_HOUR
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# ========== DEVELOPMENT SERVER ==========

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.deployment.api:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD,
        log_level=config.LOG_LEVEL.lower()
    )