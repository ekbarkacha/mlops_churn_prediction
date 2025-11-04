"""
Unit tests for modeling module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.modeling.model_utils import split_data, evaluate_model


class TestDataSplit:
    """Test suite for data splitting"""
    
    def test_split_data_basic(self):
        """Test basic data splitting"""
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200),
            'Churn': [0, 1] * 50
        })
        
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
        
        # Check sizes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_split_preserves_distribution(self):
        """Test that split preserves class distribution (stratification)"""
        df = pd.DataFrame({
            'feature1': range(100),
            'Churn': [0] * 73 + [1] * 27  # 73/27 split
        })
        
        X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
        
        # Calculate proportions
        train_prop = y_train.sum() / len(y_train)
        test_prop = y_test.sum() / len(y_test)
        original_prop = 27 / 100
        
        # Should be approximately equal (within 5%)
        assert abs(train_prop - original_prop) < 0.05
        assert abs(test_prop - original_prop) < 0.05


class TestModelEvaluation:
    """Test suite for model evaluation"""
    
    def test_evaluate_model_perfect(self):
        """Test evaluation with perfect predictions"""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        y_pred_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.1])
        
        metrics = evaluate_model(y_true, y_pred, y_pred_prob)
        
        # Perfect predictions
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
    
    def test_evaluate_model_realistic(self):
        """Test evaluation with realistic predictions"""
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 1])  # 2 errors
        y_pred_prob = np.array([0.2, 0.8, 0.3, 0.9, 0.1, 0.4, 0.6, 0.7])
        
        metrics = evaluate_model(y_true, y_pred, y_pred_prob)
        
        # Check metrics are reasonable
        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['precision'] <= 1.0
        assert 0.0 <= metrics['recall'] <= 1.0
        assert 0.0 <= metrics['f1'] <= 1.0
        assert 0.0 <= metrics['roc_auc'] <= 1.0
    
    def test_evaluate_model_returns_dict(self):
        """Test that evaluate_model returns correct structure"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        y_pred_prob = np.array([0.1, 0.9, 0.2, 0.4])
        
        metrics = evaluate_model(y_true, y_pred, y_pred_prob)
        
        # Check structure
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])