"""
Module: services/monitoring_service.py
========================================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Service for monitoring, data drift detection, and metrics reporting.
"""
import os
import pandas as pd
from typing import Optional
from src.app.config import INFERENCE_DATA_PATH, RAW_DATA_PATH, EXPECTED_COLUMNS
from src.monitoring.metrics import (
    DRIFT_SHARE_GAUGE,
    DRIFT_COUNT_GAUGE,
    PIPELINE_ERROR_COUNTER,
    MODEL_ACCURACY_GAUGE,
    MODEL_PRECISION_GAUGE,
    MODEL_RECALL_GAUGE,
    MODEL_ROC_AUC_GAUGE,
    MODEL_F1_GAUGE
)
from src.utils.logger import get_logger

logger = get_logger("customer_churn_api")


class MonitoringService:
    """Service for monitoring and metrics."""

    @staticmethod
    def read_csv_safe(path: str) -> pd.DataFrame:
        """
        Safely read a CSV file.

        Args:
            path: Path to CSV file.

        Returns:
            Dataframe or empty dataframe if file doesn't exist.
        """
        if not os.path.exists(path) or os.stat(path).st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)

    @staticmethod
    def file_exists(path: str) -> bool:
        """
        Check if file exists.

        Args:
            path: File path.

        Returns:
            True if file exists, False otherwise.
        """
        return os.path.exists(path)

    def get_feedback_data(self, limit: int = 50) -> pd.DataFrame:
        """
        Get recent inference data for feedback.

        Args:
            limit: Maximum number of records to return.

        Returns:
            Dataframe of recent inference records.
        """
        df = self.read_csv_safe(INFERENCE_DATA_PATH)
        if df.empty:
            return df

        df = df.sort_values("timestamp", ascending=False).head(limit)
        return df

    def merge_feedback(self, updated_df: pd.DataFrame) -> dict:
        """
        Merge feedback data with existing inference data.

        Args:
            updated_df: Feedback dataframe with customerID.

        Returns:
            Success message with record counts.
        """
        existing_df = self.read_csv_safe(INFERENCE_DATA_PATH)

        if existing_df.empty:
            updated_df.to_csv(INFERENCE_DATA_PATH, index=False)
            return {"message": "Feedback file saved as new inference data."}

        merged_df = pd.concat([existing_df, updated_df], ignore_index=True)
        merged_df.drop_duplicates(subset=["customerID"], keep="last", inplace=True)

        merged_df.to_csv(INFERENCE_DATA_PATH, index=False)
        logger.info("Feedback data merged successfully.")

        return {
            "message": "Feedback merged successfully with existing data.",
            "total_records": len(merged_df),
            "updated_records": len(updated_df)
        }

    def update_drift_metrics(self, model_name: str, drift_share: float, drift_count: int) -> None:
        """
        Update Prometheus drift metrics.

        Args:
            model_name: Name of the model.
            drift_share: Share of drifted features.
            drift_count: Count of drifted features.
        """
        DRIFT_SHARE_GAUGE.labels(model_name=model_name).set(drift_share)
        DRIFT_COUNT_GAUGE.labels(model_name=model_name).set(drift_count)
        logger.info(f"Drift metrics updated for {model_name}")

    def update_pipeline_error(self, pipeline_name: str) -> None:
        """
        Increment pipeline error counter.

        Args:
            pipeline_name: Name of the pipeline.
        """
        PIPELINE_ERROR_COUNTER.labels(pipeline_name=pipeline_name).inc()
        logger.warning(f"Pipeline error recorded for {pipeline_name}")

    def update_model_metrics(
        self,
        model_name: str,
        version: str,
        model_type: str,
        metrics: dict
    ) -> None:
        """
        Update model performance metrics.

        Args:
            model_name: Name of the model.
            version: Model version.
            model_type: Type of model.
            metrics: Dictionary of metric values.
        """
        MODEL_ACCURACY_GAUGE.labels(
            model_name=model_name,
            model_type=model_type,
            version=version
        ).set(metrics["accuracy"])

        MODEL_PRECISION_GAUGE.labels(
            model_name=model_name,
            model_type=model_type,
            version=version
        ).set(metrics["precision"])

        MODEL_RECALL_GAUGE.labels(
            model_name=model_name,
            model_type=model_type,
            version=version
        ).set(metrics["recall"])

        MODEL_ROC_AUC_GAUGE.labels(
            model_name=model_name,
            model_type=model_type,
            version=version
        ).set(metrics["roc_auc"])

        MODEL_F1_GAUGE.labels(
            model_name=model_name,
            model_type=model_type,
            version=version
        ).set(metrics["f1"])

        logger.info(f"Model metrics updated for {model_name} v{version}")
