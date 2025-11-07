"""
Module: services/prediction_service.py
=======================================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Service for handling model predictions and logging inference results.
"""
import os
import time
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Tuple
from src.app.config import INFERENCE_DATA_PATH, EXPECTED_COLUMNS
from src.app.ml.model_wrapper import UniversalMLflowWrapper
from src.data_pipeline.data_preprocessing import validate_schema
from src.monitoring.metrics import PREDICTION_COUNT, PREDICTION_LATENCY, PREDICTION_ERRORS
from src.app.config import MLFLOW_EXPERIMENT_NAME
from src.utils.logger import get_logger

logger = get_logger("customer_churn_api")


class PredictionService:
    """Service for handling predictions."""

    @staticmethod
    def label_churn(pred: int) -> str:
        """
        Convert numeric prediction to 'Yes'/'No' label.

        Args:
            pred: Numeric prediction (0 or 1).

        Returns:
            'Yes' for 1, 'No' for 0.
        """
        return "Yes" if int(pred) == 1 else "No"

    def validate_input(self, df: pd.DataFrame) -> None:
        """
        Validate input schema against expected columns.

        Args:
            df: Input dataframe to validate.

        Raises:
            Exception: If validation fails.
        """
        validate_schema(df, EXPECTED_COLUMNS[:-1])

    def make_predictions(
        self,
        model: UniversalMLflowWrapper,
        df: pd.DataFrame,
        include_shap: bool = False
    ) -> Tuple[List[int], List[float], Dict]:
        """
        Make predictions using the provided model.

        Args:
            model: Model wrapper to use for predictions.
            df: Input dataframe with features.
            include_shap: Whether to include SHAP explanations.

        Returns:
            Tuple of (predictions, probabilities, shap_values).
        """
        df = df[EXPECTED_COLUMNS[:-1]]
        preds, probs = model.predict(model_input=df, return_proba=True, both=True)

        shap_values = None
        if include_shap:
            shap_values = model.explain_json(df)

        return preds, probs, shap_values

    def log_inference(
        self,
        raw_input: dict,
        prediction: int,
        probability: float,
        stage: str,
        version: str,
        latency: float
    ) -> None:
        """
        Log a single inference to CSV.

        Args:
            raw_input: Raw input features.
            prediction: Predicted class (0/1).
            probability: Probability of positive class.
            stage: Model stage (Production/Staging).
            version: Model version.
            latency: Time taken for prediction in seconds.
        """
        record = raw_input.copy()
        record['Churn'] = self.label_churn(prediction)
        record['probability'] = probability
        record['timestamp'] = datetime.now(timezone.utc).isoformat()
        record['model_stage'] = stage
        record['model_version'] = version
        record["latency"] = latency

        df = pd.DataFrame([record])
        os.makedirs(os.path.dirname(INFERENCE_DATA_PATH), exist_ok=True)
        file_exists = os.path.exists(INFERENCE_DATA_PATH)
        df.to_csv(INFERENCE_DATA_PATH, mode='a', header=not file_exists, index=False)

    def log_predictions_batch(
        self,
        raw_inputs: List[dict],
        preds: List[int],
        probs: List[float],
        stage: str,
        version: str,
        latency: float
    ) -> None:
        """
        Log batch predictions in background.

        Args:
            raw_inputs: List of raw input dictionaries.
            preds: List of predictions.
            probs: List of probabilities.
            stage: Model stage.
            version: Model version.
            latency: Prediction latency.
        """
        for i, pred in enumerate(preds):
            self.log_inference(
                raw_input=raw_inputs[i],
                prediction=float(pred),
                probability=probs[i],
                stage=stage,
                version=version,
                latency=latency
            )

    def record_metrics(
        self,
        stage: str,
        version: str,
        duration: float,
        error: bool = False
    ) -> None:
        """
        Record Prometheus metrics for predictions.

        Args:
            stage: Model stage.
            version: Model version.
            duration: Prediction duration.
            error: Whether an error occurred.
        """
        if error:
            PREDICTION_ERRORS.labels(
                model_name=MLFLOW_EXPERIMENT_NAME,
                version=version,
                stage=stage
            ).inc()
        else:
            PREDICTION_COUNT.labels(
                model_name=MLFLOW_EXPERIMENT_NAME,
                version=version,
                stage=stage
            ).inc()
            PREDICTION_LATENCY.labels(
                model_name=MLFLOW_EXPERIMENT_NAME,
                version=version,
                stage=stage
            ).observe(duration)
