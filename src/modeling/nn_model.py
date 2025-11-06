"""
Module: nn_model.py 
=======================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description: 
------------
This module defines a simple feed-forward neural network built using **PyTorch**, 
along with a custom **MLflow PyFunc wrapper** for model serving and deployment.

It includes:
- `NN`: A fully connected neural network for binary classification tasks.
- `PyTorchWrapper`: A class that integrates the PyTorch model with MLflow’s 
  PyFunc API, allowing inference from a Pandas DataFrame in production environments.

Key Features:
-------------
- Compatible with MLflow model registry and deployment workflows.
- Provides both class predictions (`predict`) and probability scores (`predict_proba`).
- Automatically applies a sigmoid activation for binary outputs.
"""
# Imports
import numpy as np
import mlflow
import pandas as pd
import torch
import torch.nn as nn

# Neural Network Model
class NN(nn.Module):
    def __init__(self, input_dim, hidden_units):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1)  # Return logits
        )

    def forward(self, x):
        return self.net(x)

# PyTorchWrapper
class PyTorchWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model: torch.nn.Module, device=None, threshold=0.5):
        self.model = model
        self.device = torch.device("cpu")

        self.threshold = threshold
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context, model_input: pd.DataFrame):
        """
        Takes a pandas DataFrame and returns predictions.
        Returns probabilities if available, else binary classes.
        """
        with torch.no_grad():
            x = torch.tensor(model_input.values, dtype=torch.float32, device=self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= self.threshold).astype(int)
        return preds
    
    def predict_proba(self, context, model_input: pd.DataFrame):
        """
        Predict probability scores for a given dataframe.
        Returns probabilities between 0 and 1.
        """
        with torch.no_grad():
            x = torch.tensor(model_input.values, dtype=torch.float32, device=self.device)
            logits = self.model(x)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs
    

