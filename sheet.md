# ============================================================================
# PRODUCTION MODEL CONFIGURATION
# Best Neural Network: F1=64.51%, ROC_AUC=85.35%
# ============================================================================

# Production Neural Network (PyTorch)
# This is the BEST performing model after extensive testing
neural_net:
  hidden_units: 128   # plusieurs couches
  lr: 0.001
  epochs: 60
  batch_size: 256
  dropout: 0.3
  activation: "relu"
  test_size: 0.2
  random_state: 2
  use_smote: True
  smote_strategy: 1
 

  # Performance metrics (test set):
  # - F1: 64.51%
  # - Precision: 58.48%
  # - Recall: 71.93%
  # - ROC AUC: 85.35%
  # - Accuracy: 78.99%

# Random Forest Classifier (Backup model)
random_forest:
  n_estimators: 300
  max_depth: 25
  min_samples_split: 5
  random_state: 42
  class_weight: "balanced"
  tuning:
    enabled: false  # Disabled for production

# XGBoost Classifier (Backup model)
xgboost:
  learning_rate: 0.03
  max_depth: 7
  n_estimators: 600
  scale_pos_weight: 3.0
  min_child_weight: 1
  random_state: 42
  tuning:
    enabled: false  # Disabled for production

# Training Configuration
training:
  test_size: 0.2
  random_state: 2
  use_smote: true
  smote_random_state: 2
  smote_strategy: 1.0  # 50/50 balance (optimal)

# Production Settings
production:
  model_name: "churn_prediction_neural_net"
  model_version: "1.0.0"
  threshold: 0.5  # Classification threshold (can be tuned)
  feature_file: "telco_churn_features_advanced.csv"
  n_features: 42

seed: 2



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
