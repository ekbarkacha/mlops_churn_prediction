"""
Neural Network Model - PyTorch implementation
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import mlflow



# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger

logger = get_logger(__name__)

    

class NN(nn.Module):
    """
    Simple Neural Network for binary classification with dropout.
    Architecture: Input -> Hidden -> Dropout -> Output
    """
    
    def __init__(self, input_dim: int, hidden_units: int = 128, dropout: float = 0.3):
        """
        Initialize Neural Network.
        
        Args:
            input_dim (int): Number of input features
            hidden_units (int): Number of hidden units
            dropout (float): Dropout probability for regularization
        """
        super(NN, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_units, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
        
        Returns:
            torch.Tensor: Output logits
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x




class PyTorchWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for PyTorch model to make it MLflow-compatible.
    """
    
    def __init__(self, model: nn.Module, device: str = None, threshold: float = 0.5):
        """
        Initialize wrapper.
        
        Args:
            model (nn.Module): PyTorch model
            device (str): Device to use ('cpu', 'cuda', 'mps')
            threshold (float): Classification threshold
        """
        self.model = model
        
        # Auto-detect device
        if device:
            self.device = torch.device(device)
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Apple Silicon
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")  # NVIDIA GPU
        else:
            self.device = torch.device("cpu")
        
        self.threshold = threshold
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"PyTorchWrapper initialized on device: {self.device}")
    
    def predict(self, context, model_input):
        """
        Predict class labels.
        
        Args:
            context: MLflow context (unused)
            model_input: Input features (DataFrame, numpy array, or list)
        
        Returns:
            np.ndarray: Predicted labels (0 or 1)
        """
        with torch.no_grad():
            # Convert input to numpy array (handle multiple input types)
            if isinstance(model_input, pd.DataFrame):
                x_numpy = model_input.values
            elif isinstance(model_input, list):
                x_numpy = np.array(model_input)
            elif isinstance(model_input, np.ndarray):
                x_numpy = model_input
            else:
                raise ValueError(f"Unsupported input type: {type(model_input)}")
            
            # Convert to tensor
            x = torch.tensor(
                x_numpy, 
                dtype=torch.float32, 
                device=self.device
            )
            
            # Forward pass
            logits = self.model(x)
            
            # Convert to probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # Apply threshold
            preds = (probs >= self.threshold).astype(int)
        
        return preds
    
    def predict_proba(self, context, model_input):
        """
        Predict class probabilities.
        
        Args:
            context: MLflow context (unused)
            model_input: Input features (DataFrame, numpy array, or list)
        
        Returns:
            np.ndarray: Predicted probabilities
        """
        with torch.no_grad():
            # Convert input to numpy array (handle multiple input types)
            if isinstance(model_input, pd.DataFrame):
                x_numpy = model_input.values
            elif isinstance(model_input, list):
                x_numpy = np.array(model_input)
            elif isinstance(model_input, np.ndarray):
                x_numpy = model_input
            else:
                raise ValueError(f"Unsupported input type: {type(model_input)}")
            
            # Convert to tensor
            x = torch.tensor(
                x_numpy,
                dtype=torch.float32,
                device=self.device
            )
            
            # Forward pass
            logits = self.model(x)
            
            # Convert to probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        return probs