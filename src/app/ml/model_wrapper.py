"""
Module: ml/model_wrapper.py
============================

Author: AIMS-AMMI STUDENT
Created: November 2025
Description:
------------
Universal MLflow Model Wrapper for multiple ML frameworks.

This module provides the `UniversalMLflowWrapper` class which standardizes prediction,
probability estimation, and explainability across different types of machine learning models:
- Scikit-learn
- XGBoost
- PyTorch Neural Networks

Key Features:
- Unified API for `predict`, `predict_proba`, and SHAP-based `explain`.
- Automatic preprocessing using centralized InferencePreprocessor.
- Automatic device management for PyTorch models (CPU, CUDA, MPS).
- SHAP explainability initialization and caching.
"""
import mlflow.pyfunc
import torch
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import xgboost as xgb
import shap
from src.app.ml.preprocessing import InferencePreprocessor


class UniversalMLflowWrapper(mlflow.pyfunc.PythonModel):
    """
    This class standardizes prediction and explainability behavior across multiple ML model types.
    It automatically detects and adapts to Scikit-learn, XGBoost, or Neural Network models.

    Key Features:
    - Unified API for `predict`, `predict_proba`, and SHAP-based `explainability`.
    - Automatic preprocessing via InferencePreprocessor.
    - Background SHAP explainer caching for performance.
    """

    def __init__(self, model, threshold=0.5, version_dir=None, background_data=None, device=None):
        """
        Initialize the model wrapper.

        Args:
            model: The ML model (sklearn, xgboost, or pytorch).
            threshold: Classification threshold for binary predictions.
            version_dir: Directory containing model version and preprocessors.
            background_data: Background data for SHAP explainer.
            device: Device for PyTorch models (optional).
        """
        self.model = model
        self.threshold = threshold
        self.preprocessor = InferencePreprocessor(version_dir) if version_dir else None
        self.explainer = None

        # Auto detect model type
        if isinstance(model, BaseEstimator):
            self.model_type = "sklearn"
        elif isinstance(model, xgb.Booster) or isinstance(model, xgb.XGBClassifier):
            self.model_type = "xgboost"
        elif isinstance(model, torch.nn.Module):
            self.model_type = "nn"
            # Setup device
            if device:
                self.device = device
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            self.model.to(self.device)
            self.model.eval()
        else:
            self.model_type = "unknown"

        # Initialize SHAP explainer if background data provided
        if background_data is not None and isinstance(background_data, pd.DataFrame) and self.preprocessor:
            self._init_explainer(self.preprocessor.preprocess(background_data))

    def predict(self, model_input: pd.DataFrame, return_proba=False, both=False, context=None):
        """
        Make predictions on the input data.

        Args:
            model_input: Input dataframe with raw features.
            return_proba: If True, return probabilities instead of labels.
            both: If True, return both labels and probabilities as a tuple.
            context: Optional MLflow context (unused).

        Returns:
            Predictions as labels or probabilities.
        """
        # Preprocess input
        if self.preprocessor:
            model_input = self.preprocessor.preprocess(model_input)

        # Get probabilities based on model type
        if self.model_type == "nn":
            with torch.no_grad():
                x = torch.tensor(model_input.values, dtype=torch.float32, device=self.device)
                logits = self.model(x)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
        elif self.model_type == "sklearn":
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(model_input)[:, 1]
            else:
                probs = self.model.predict(model_input)
        elif self.model_type == "xgboost":
            if isinstance(self.model, xgb.XGBClassifier):
                probs = self.model.predict_proba(model_input)[:, 1]
            else:
                dmatrix = xgb.DMatrix(model_input)
                probs = self.model.predict(dmatrix)
        else:
            raise ValueError(f"Cannot predict: unknown model type {self.model_type}")

        # Return based on requested format
        if return_proba and both:
            return (probs >= self.threshold).astype(int), probs
        elif return_proba:
            return probs

        return (probs >= self.threshold).astype(int)

    def predict_proba(self, model_input: pd.DataFrame, context=None):
        """
        Get prediction probabilities.

        Args:
            model_input: Input dataframe with raw features.
            context: Optional MLflow context (unused).

        Returns:
            Array of probabilities.
        """
        return self.predict(model_input, return_proba=True, context=context)

    def _init_explainer(self, background_data: pd.DataFrame):
        """
        Initialize SHAP explainer with preprocessed background data.

        Args:
            background_data: Preprocessed background data for SHAP.
        """
        if self.model_type in ["sklearn", "xgboost"]:
            if hasattr(self.model, "predict_proba"):
                f = lambda X: self.model.predict_proba(X)[:, 1]
            else:
                f = self.model.predict
            self.explainer = shap.Explainer(f, background_data)
        elif self.model_type == "nn":
            def f(X):
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    logits = self.model(X_tensor)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                return probs
            self.explainer = shap.KernelExplainer(f, background_data)
        else:
            raise ValueError(f"Cannot init SHAP: unknown model type {self.model_type}")

    def explain(self, model_input: pd.DataFrame, nsamples=100):
        """
        Generate SHAP explanations for the input data.

        Args:
            model_input: Input dataframe (will be preprocessed).
            nsamples: Number of samples for SHAP estimation.

        Returns:
            SHAP values.
        """
        if self.explainer is None:
            # Create background from input if no explainer exists
            background = (
                pd.concat([model_input] * 50, ignore_index=True)
                if model_input.shape[0] == 1
                else model_input
            )
            if self.preprocessor:
                background = self.preprocessor.preprocess(background)
            self._init_explainer(background)

        # Preprocess input before explaining
        if self.preprocessor:
            model_input = self.preprocessor.preprocess(model_input)

        shap_values = self.explainer(model_input)
        return shap_values

    def explain_json(self, model_input: pd.DataFrame, nsamples=100):
        """
        Return SHAP explanations as a JSON-safe dictionary.

        Args:
            model_input: Input dataframe (will be preprocessed).
            nsamples: Number of samples for SHAP estimation.

        Returns:
            Dictionary with SHAP values in JSON-safe format.
        """
        shap_values = self.explain(model_input, nsamples=nsamples)

        if hasattr(shap_values, "values"):
            values = shap_values.values.tolist()
            base_values = (
                shap_values.base_values.tolist()
                if hasattr(shap_values, "base_values")
                else None
            )
        else:
            values = shap_values.tolist()
            base_values = None

        # Get column names from preprocessed input
        if self.preprocessor:
            preprocessed_input = self.preprocessor.preprocess(model_input)
            columns = preprocessed_input.columns.tolist()
        else:
            columns = model_input.columns.tolist()

        return {
            "summary": {
                "mean_abs_shap": np.mean(np.abs(values), axis=0).tolist(),
                "max_abs_shap": np.max(np.abs(values), axis=0).tolist(),
            },
            "per_feature": dict(zip(columns, np.mean(np.abs(values), axis=0).tolist())),
            "raw": {
                "values": values,
                "base_values": base_values,
                "columns": columns,
            },
        }
