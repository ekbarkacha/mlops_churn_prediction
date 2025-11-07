"""
Module: prediction.py
======================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Prediction-related Pydantic schemas for the churn prediction API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class CustomerInput(BaseModel):
    """Schema for customer input data."""
    customerID: str
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionResult(BaseModel):
    """Schema for a single prediction result."""
    prediction: int
    probability: float


class AdminPredictionResponse(BaseModel):
    """Schema for admin prediction response with full details."""
    role: str
    results: List[Dict[str, Any]]
    shap_values: Optional[Dict[str, Any]] = None


class AgentPredictionResponse(BaseModel):
    """Schema for agent prediction response."""
    role: str
    prediction: int
    probability: float


class ExplainabilityRequest(BaseModel):
    """Schema for explainability request."""
    pass  # Uses CustomerInput or file upload


class ExplainabilityResponse(BaseModel):
    """Schema for explainability response."""
    role: str
    results: List[Dict[str, Any]]
    shap_values: Dict[str, Any]


class FeedbackDataResponse(BaseModel):
    """Schema for feedback data response."""
    pass  # Returns list of dictionaries


class UploadDataResponse(BaseModel):
    """Schema for data upload response."""
    message: str
    file_saved_at: str
    rows: int
    columns: int


class CanaryPercentageRequest(BaseModel):
    """Schema for canary percentage update."""
    percentage: float = Field(..., ge=0.0, le=1.0)


class CanaryPercentageResponse(BaseModel):
    """Schema for canary percentage response."""
    message: str
