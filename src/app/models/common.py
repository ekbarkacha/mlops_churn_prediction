"""
Module: common.py
==================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Common Pydantic schemas used across the application.
"""
from pydantic import BaseModel
from typing import Optional, Any, Dict


class SuccessResponse(BaseModel):
    """Generic success response."""
    message: str
    status: str = "success"


class ErrorResponse(BaseModel):
    """Generic error response."""
    detail: str
    status: str = "error"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    model_status: Dict[str, bool]
    timestamp: str


class MessageResponse(BaseModel):
    """Simple message response."""
    msg: str
