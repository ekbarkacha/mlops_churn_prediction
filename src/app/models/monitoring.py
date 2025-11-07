"""
Module: monitoring.py
======================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Monitoring-related Pydantic schemas for metrics and drift detection.
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any


class DriftMetricsUpdate(BaseModel):
    """Schema for drift metrics update."""
    from_: str = Field(..., alias="from")
    model_name: str
    drift_share: float = Field(..., ge=0.0, le=1.0)
    drift_count: int = Field(..., ge=0)

    class Config:
        populate_by_name = True


class PipelineErrorUpdate(BaseModel):
    """Schema for pipeline error update."""
    from_: str = Field(..., alias="from")
    pipeline_name: str

    class Config:
        populate_by_name = True


class ModelMetrics(BaseModel):
    """Schema for model performance metrics."""
    accuracy: float = Field(..., ge=0.0, le=1.0)
    precision: float = Field(..., ge=0.0, le=1.0)
    recall: float = Field(..., ge=0.0, le=1.0)
    roc_auc: float = Field(..., ge=0.0, le=1.0)
    f1: float = Field(..., ge=0.0, le=1.0)


class ModelDecayUpdate(BaseModel):
    """Schema for model decay metrics update."""
    from_: str = Field(..., alias="from")
    model_name: str
    version: str
    model_type: str
    metric: ModelMetrics

    class Config:
        populate_by_name = True


class MetricsUpdateRequest(BaseModel):
    """Generic metrics update request."""
    from_: str = Field(..., alias="from")

    class Config:
        populate_by_name = True
        extra = "allow"


class MetricsUpdateResponse(BaseModel):
    """Schema for metrics update response."""
    status: str
    source: str


class ModelVersionMetrics(BaseModel):
    """Schema for a single model version metrics."""
    run_id: str
    metrics: Dict[str, float]
    name: str


class AllVersionMetricsResponse(BaseModel):
    """Schema for all version metrics response."""
    pass  # Returns dict of version -> metrics
