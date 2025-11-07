"""
Module: core/middleware.py
===========================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Custom middleware for the FastAPI application including:
- Request/response logging
- Performance monitoring
- CORS configuration
"""
import time
import uuid
import json
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from src.utils.logger import get_logger
from src.monitoring.metrics import API_REQUEST_COUNT, API_LATENCY, API_ERRORS

logger = get_logger("customer_churn_api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log and monitor all HTTP requests.
    Excludes certain internal endpoints like /metrics.
    """

    EXCLUDED_PATHS = ["/metrics", "/verify-session", "/update_metrics", "/iframe_data_monitoring_proxy"]

    async def dispatch(self, request: Request, call_next):
        """
        Process each request, log metadata, and update Prometheus metrics.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware or route handler.

        Returns:
            HTTP response with added headers.
        """
        start = time.perf_counter()
        method = request.method
        endpoint = request.url.path

        # Skip internal endpoints
        if endpoint in self.EXCLUDED_PATHS:
            return await call_next(request)

        try:
            # Generate or get request ID
            req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

            # Process request
            response = await call_next(request)

            duration = time.perf_counter() - start
            response.headers["X-Process-Time"] = f"{duration:.3f}s"
            response.headers["X-Request-ID"] = req_id

            # Log request metadata
            log_data = {
                "id": req_id,
                "method": method,
                "path": endpoint,
                "status": response.status_code,
                "duration": round(duration, 3)
            }

            logger.info(json.dumps(log_data))

            # Update Prometheus metrics
            API_REQUEST_COUNT.labels(endpoint=endpoint, method=method).inc()
            API_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

            return response

        except Exception as e:
            API_ERRORS.labels(endpoint=endpoint, method=method, status_code=500).inc()
            logger.exception(f"{method} {endpoint} failed: {str(e)}")
            return Response("Internal server error", status_code=500)


def setup_cors(app):
    """
    Configure CORS middleware to allow frontend requests.

    Args:
        app: FastAPI application instance.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Update with specific frontend domain in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
