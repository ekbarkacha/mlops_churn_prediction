"""
Module: model_wrapper.py
========================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
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
- Automatic preprocessing (encoding, scaling, feature selection) using stored preprocessors.
- Automatic device management for PyTorch models (CPU, CUDA, MPS).
- SHAP explainability initialization and caching.
"""
# Imports and setup
import mlflow.pyfunc
import torch
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import xgboost as xgb
import shap
import os
import joblib
# Custom utility import
from src.utils.const import scaler_file_name,label_encoders_file_name

class UniversalMLflowWrapper(mlflow.pyfunc.PythonModel):
    """
    This class standardizes prediction and explainability behavior across multiple ML model types.
    It automatically detects and adapts to Scikit-learn, XGBoost, or Neural Network models.

    Key Features:
    - Unified API for `predict`, `predict_proba`, and SHAP-based `explainability`.
    - Automatic preprocessing (encoding, scaling, feature selection).
    - Background SHAP explainer caching for performance.
    """

    def __init__(self, model, threshold=0.5,version_dir=None,background_data=None, device=None):
        self.model = model
        self.threshold = threshold
        self.preprocessors_path = os.path.join(version_dir, "preprocessors")
        self.scaler = joblib.load(os.path.join(self.preprocessors_path, scaler_file_name))
        self.encoders = joblib.load(os.path.join(self.preprocessors_path, label_encoders_file_name))
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
            #print(f" Warning: Unknown model type detected: {type(model)}")

        if background_data is not None and isinstance(background_data, pd.DataFrame):
            self._init_explainer(self.preprocess(background_data))


    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply same preprocessing as in training:
        - data cleaning
        - encoding categorical
        - feature selection
        - scaling
        """

        ######## Preprocessing #######
        # Cleaning
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        # Encode categorical columns
        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].map(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)

        # Feature_creation

        # Feature Transformation

        # Feature selection
        drop_features = ['customerID','PhoneService', 'gender', 'StreamingTV', 'StreamingMovies', 'MultipleLines', 'InternetService']
        df.drop(columns=drop_features, inplace=True, errors='ignore')

        # Feature Scaling (numeric columns)
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        df[num_cols] = self.scaler.transform(df[num_cols])

        return df
    def predict(self, model_input: pd.DataFrame, return_proba=False, both=False, context=None):
        """Return labels or probabilities depending on return_proba."""

        model_input = self.preprocess(model_input)
        
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

        if return_proba and both:
            return (probs >= self.threshold).astype(int),probs
        elif return_proba:
            return probs

        return (probs >= self.threshold).astype(int)

    def predict_proba(self, model_input: pd.DataFrame, context=None):
        return self.predict(model_input, return_proba=True, context=context)

    
    def _init_explainer(self, background_data: pd.DataFrame):
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
        if self.explainer is None:
            background = (
                pd.concat([model_input] * 50, ignore_index=True)
                if model_input.shape[0] == 1
                else model_input
            )
            self._init_explainer(background)

        shap_values = self.explainer(model_input)
        return shap_values

    def explain_json(self, model_input: pd.DataFrame, nsamples=100):
        """
        Return SHAP explanations as a JSON-safe dictionary. Handles both SHAP Explanation objects and raw numpy arrays.
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

        return {
            "summary": {
                "mean_abs_shap": np.mean(np.abs(values), axis=0).tolist(),
                "max_abs_shap": np.max(np.abs(values), axis=0).tolist(),
            },
            "per_feature": dict(zip(model_input.columns.tolist(), np.mean(np.abs(values), axis=0).tolist())),
            "raw": {
                "values": values,
                "base_values": base_values,
                "columns": model_input.columns.tolist(),
            },
        }

