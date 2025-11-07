"""
Module: services/model_service.py
===================================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Service for loading, managing, and selecting ML models from MLflow.
"""
import os
import random
import mlflow.pyfunc
import pandas as pd
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from typing import Dict, Tuple, Optional
from src.app.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_DIR,
    EXPECTED_COLUMNS,
    RAW_DATA_PATH
)
from src.app.ml.model_wrapper import UniversalMLflowWrapper
from src.monitoring.metrics import CURRENT_MODEL_VERSION
from src.utils.logger import get_logger

logger = get_logger("customer_churn_api")


class ModelService:
    """Service for managing ML models."""

    def __init__(self):
        self.models: Dict[str, UniversalMLflowWrapper] = {}
        self.version_folders: Dict[str, str] = {}
        self.canary_weight: float = 0.05
        self.model_metrics_cache: Dict = {}

    async def load_models(self) -> None:
        """
        Load models from MLflow registry for Production and Staging stages.
        Downloads and caches models locally if not already cached.
        """
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        logger.info(f"Connected to MLflow at {MLFLOW_TRACKING_URI}")

        stages = ["Production", "Staging"]
        background = None

        for stage in stages:
            try:
                # Get latest model version for this stage
                versions = client.get_latest_versions(name=MLFLOW_EXPERIMENT_NAME, stages=[stage])
                if not versions:
                    logger.warning(f"No model found for stage '{stage}'")
                    continue

                # Sort and select latest version
                latest_version = sorted(versions, key=lambda v: int(v.version))[-1]
                model_uri = f"models:/{latest_version.name}/{latest_version.version}"
                version_folder = f"{MODEL_DIR}/v{latest_version.version}"

                # Get run name
                run_info = client.get_run(latest_version.run_id)
                run_name = run_info.data.tags.get("mlflow.runName", "Unnamed Run")

                os.makedirs(MODEL_DIR, exist_ok=True)

                # Download and cache model if not cached
                if not os.path.exists(version_folder):
                    logger.info(f"Downloading {stage} model version {latest_version.version}...")
                    model = mlflow.pyfunc.load_model(model_uri)

                    mlflow.pyfunc.save_model(
                        path=version_folder,
                        python_model=model._model_impl.python_model
                    )

                    logger.info(f"Downloading preprocessor artifacts for run_id: {latest_version.run_id}")
                    client.download_artifacts(
                        run_id=latest_version.run_id,
                        path="preprocessors",
                        dst_path=version_folder
                    )

                    logger.info(f"Model version {latest_version.version} cached at {version_folder}.")
                else:
                    logger.info(f"Loading {stage} cached model version {latest_version.version}")
                    model = mlflow.pyfunc.load_model(version_folder)

                # Load background data for SHAP
                if background is None:
                    try:
                        background = pd.read_csv(RAW_DATA_PATH)
                        background = background[EXPECTED_COLUMNS[:-1]].sample(100, random_state=42)
                    except FileNotFoundError:
                        logger.warning(f"Background data not found at {RAW_DATA_PATH}.")

                # Wrap model
                self.models[stage] = UniversalMLflowWrapper(
                    model._model_impl.python_model.model,
                    version_dir=version_folder,
                    background_data=background
                )

                self.version_folders[stage] = version_folder

                CURRENT_MODEL_VERSION.labels(
                    model_name=MLFLOW_EXPERIMENT_NAME,
                    model_type=run_name,
                    stage=stage
                ).set(latest_version.version)

                logger.info(f"{stage} model {MLFLOW_EXPERIMENT_NAME} version {latest_version.version} loaded.")

            except MlflowException as e:
                logger.warning(f"Could not load model for stage {stage}: {e}")

        # Load metrics cache
        self.model_metrics_cache = await self.get_model_metrics_from_registry(MLFLOW_EXPERIMENT_NAME)
        logger.info("All models initialized successfully.")

    async def get_model_metrics_from_registry(self, model_name: str) -> dict:
        """
        Retrieve metrics for all versions of a model from MLflow registry.

        Args:
            model_name: Name of the MLflow model.

        Returns:
            Dictionary of version -> metrics.
        """
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        model_versions = client.search_model_versions(f"name='{model_name}'")

        results = {}
        for mv in model_versions:
            run = client.get_run(mv.run_id)
            results[f"v{mv.version}"] = {
                "run_id": mv.run_id,
                "metrics": run.data.metrics,
                "name": run.data.tags.get("mlflow.runName", "Unnamed Run")
            }

        return results

    def get_model_for_inference(self) -> Tuple[Optional[UniversalMLflowWrapper], Optional[str], Optional[str]]:
        """
        Select a model for inference based on canary deployment logic.

        Returns:
            Tuple of (model, stage, version).
        """
        if not self.models:
            return None, None, None

        rand_val = random.random()
        if "Staging" in self.models and rand_val < self.canary_weight:
            stage = "Staging"
        else:
            stage = "Production" if "Production" in self.models else next(iter(self.models.keys()))

        model = self.models[stage]
        version = self.version_folders[stage].split("/")[-1]

        return model, stage, version

    def set_canary_weight(self, percentage: float) -> None:
        """
        Set the canary deployment weight.

        Args:
            percentage: Canary weight (0.0 to 1.0).
        """
        self.canary_weight = percentage
        logger.info(f"Canary weight set to {percentage * 100:.0f}%")

    def cleanup(self) -> None:
        """Release model resources."""
        self.models.clear()
        self.version_folders.clear()
        self.model_metrics_cache.clear()
        logger.info("Model resources released.")
