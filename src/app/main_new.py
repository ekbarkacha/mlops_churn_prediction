"""
Module: main.py
================

Author: AIMS-AMMI STUDENT
Created: November 2025 (Refactored)
Description:
------------
Main FastAPI application entry point for the Customer Churn Prediction API.

This is a refactored, minimal version that delegates responsibilities to:
- core/ for security and dependencies
- services/ for business logic
- api/v1/ for route handlers
- ml/ for machine learning operations
- models/ for Pydantic schemas
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY

# Core imports
from src.app.core.middleware import LoggingMiddleware, setup_cors

# Service imports
from src.app.services.model_service import ModelService

# API route imports
from src.app.api.v1 import auth, users, predictions, monitoring, admin

# Utilities
from src.utils.logger import get_logger

logger = get_logger("customer_churn_api")


# ============================================================================
# Application Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown lifecycle.

    On startup:
    - Initialize and load ML models from MLflow
    - Download preprocessing artifacts
    - Prepare SHAP explainers
    - Cache model metrics

    On shutdown:
    - Release model resources
    - Clean up state
    """
    logger.info("Starting up FastAPI app and initializing ML models...")

    # Initialize model service
    model_service = ModelService()
    await model_service.load_models()

    # Store model service in app state for access by routes
    app.state.model_service = model_service

    yield

    # Cleanup on shutdown
    logger.info("Shutting down app — releasing model resources...")
    model_service.cleanup()
    logger.info("App shutdown complete, models unloaded.")


# ============================================================================
# FastAPI Application Initialization
# ============================================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="MLOps API for customer churn prediction with canary deployment, monitoring, and explainability",
    version="2.0.0",
    lifespan=lifespan
)


# ============================================================================
# Middleware Configuration
# ============================================================================

# CORS middleware for frontend requests
setup_cors(app)

# Custom logging and monitoring middleware
app.add_middleware(LoggingMiddleware)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ============================================================================
# Prometheus Metrics
# ============================================================================

# Initialize Prometheus instrumentator
instrumentator = Instrumentator().instrument(app)
instrumentator.expose(app)


@app.get("/metrics", tags=["Metrics"])
def metrics():
    """
    Expose Prometheus metrics for scraping.

    Returns:
        Prometheus-formatted metrics data.
    """
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# ============================================================================
# API Routes Registration
# ============================================================================

# Authentication routes
app.include_router(auth.router)

# User management routes
app.include_router(users.router)

# Prediction routes
app.include_router(predictions.router)

# Monitoring routes
app.include_router(monitoring.router)

# Admin routes
app.include_router(admin.router)


# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/", tags=["Root"])
def root():
    """
    Root endpoint of the API.

    Returns:
        API status and version information.
    """
    return {
        "message": "Customer Churn Prediction API is running.",
        "status": "ok",
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    Health check endpoint.

    Verifies that the API and ML models are loaded and operational.

    Returns:
        Health status and model availability.
    """
    model_service = getattr(app.state, "model_service", None)

    if model_service:
        models_loaded = bool(model_service.models)
        model_status = {
            stage: bool(model)
            for stage, model in model_service.models.items()
        }
    else:
        models_loaded = False
        model_status = {}

    return {
        "status": "ok" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "model_status": model_status,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# ============================================================================
# Application Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
