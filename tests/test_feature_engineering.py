"""
Unit tests for feature engineering module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.feature_engineering import (
    feature_creation,
    feature_transformation,
    feature_selection,
    feature_scaling,
    feature_engineering_pipeline
)
from src.utils.const import (
    FEATURES_TO_DROP,
    FEATURES_TO_SCALE,
    TARGET_COLUMN,
    FINAL_FEATURES
)


class TestFeatureSelection:
    """Test suite for feature selection"""
    
    def test_feature_selection_drops_correct_columns(self):
        """Test that correct columns are dropped"""
        # Create mock dataframe with all columns
        all_cols = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        df = pd.DataFrame({col: [1, 2, 3] for col in all_cols})
        
        df_selected = feature_selection(df)
        
        # Check dropped columns are gone
        for col in FEATURES_TO_DROP:
            assert col not in df_selected.columns
        
        # Check important columns remain
        assert 'Partner' in df_selected.columns
        assert 'Contract' in df_selected.columns
        assert 'Churn' in df_selected.columns
    
    def test_feature_selection_preserves_target(self):
        """Test that target column is preserved"""
        # Create full dataframe with all expected columns
        all_cols = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        df = pd.DataFrame({col: [0, 1, 0] for col in all_cols})
        
        df_selected = feature_selection(df)
        
        # Target must be preserved
        assert TARGET_COLUMN in df_selected.columns
        # Check values unchanged
        assert df_selected[TARGET_COLUMN].tolist() == [0, 1, 0]
    
    def test_feature_selection_reduces_columns(self):
        """Test that feature selection reduces number of columns"""
        all_cols = [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ]
        df = pd.DataFrame({col: [1, 2, 3] for col in all_cols})
        
        df_selected = feature_selection(df)
        
        # Should drop 7 features: 21 - 7 = 14
        assert len(df_selected.columns) == 14


class TestFeatureScaling:
    """Test suite for feature scaling"""
    
    def test_minmax_scaling(self):
        """Test MinMaxScaler scales to [0, 1]"""
        df = pd.DataFrame({
            'tenure': [1, 12, 24, 36, 72],
            'MonthlyCharges': [20.0, 50.0, 80.0, 100.0, 120.0],
            'TotalCharges': [100.0, 500.0, 1000.0, 2000.0, 8000.0],
            'Churn': [0, 1, 0, 1, 0]
        })
        
        df_scaled = feature_scaling(df, method='minmax', save=False)
        
        # Check values are between 0 and 1
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            assert df_scaled[col].min() >= 0.0
            assert df_scaled[col].max() <= 1.0
    
    def test_standard_scaling(self):
        """Test StandardScaler centers around 0"""
        df = pd.DataFrame({
            'tenure': [1, 12, 24, 36, 72],
            'MonthlyCharges': [20.0, 50.0, 80.0, 100.0, 120.0],
            'TotalCharges': [100.0, 500.0, 1000.0, 2000.0, 8000.0],
            'Churn': [0, 1, 0, 1, 0]
        })
        
        df_scaled = feature_scaling(df, method='standard', save=False)
        
        # Check mean is approximately 0 (allowing for floating point errors)
        for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
            assert abs(df_scaled[col].mean()) < 1e-10
    
    def test_scaling_preserves_target(self):
        """Test that target column is not scaled"""
        df = pd.DataFrame({
            'tenure': [1, 12, 24],
            'MonthlyCharges': [20.0, 50.0, 80.0],
            'TotalCharges': [100.0, 500.0, 1000.0],
            'Churn': [0, 1, 0]
        })
        
        df_scaled = feature_scaling(df, method='minmax', save=False)
        
        # Target should remain unchanged
        assert df_scaled['Churn'].tolist() == [0, 1, 0]
    
    def test_invalid_scaling_method(self):
        """Test that invalid scaling method raises error"""
        df = pd.DataFrame({
            'tenure': [1, 2, 3],
            'Churn': [0, 1, 0]
        })
        
        # Should raise RuntimeError (which wraps ValueError)
        with pytest.raises(RuntimeError, match="Feature scaling failed"):
            feature_scaling(df, method='invalid', save=False)


class TestFeatureCreation:
    """Test suite for feature creation"""
    
    def test_feature_creation_returns_dataframe(self):
        """Test that feature creation returns a dataframe"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = feature_creation(df)
        assert isinstance(result, pd.DataFrame)
    
    def test_feature_creation_preserves_data(self):
        """Test that feature creation doesn't modify data (placeholder)"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        result = feature_creation(df)
        
        # Currently no features created, so should be identical
        pd.testing.assert_frame_equal(result, df)


class TestFeatureTransformation:
    """Test suite for feature transformation"""
    
    def test_feature_transformation_returns_dataframe(self):
        """Test that feature transformation returns a dataframe"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        result = feature_transformation(df)
        assert isinstance(result, pd.DataFrame)
    
    def test_feature_transformation_preserves_data(self):
        """Test that transformation doesn't modify data (placeholder)"""
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        result = feature_transformation(df)
        
        # Currently no transformations, so should be identical
        pd.testing.assert_frame_equal(result, df)


class TestFeatureEngineeringPipeline:
    """Integration test for full feature engineering pipeline"""
    
    def test_full_pipeline(self):
        """Test complete feature engineering pipeline"""
        try:
            from src.utils.const import PROCESSED_DATA_DIR, PROCESSED_FILE_NAME
            processed_path = PROCESSED_DATA_DIR / PROCESSED_FILE_NAME
            
            # Skip if preprocessed data doesn't exist
            if not processed_path.exists():
                pytest.skip("Preprocessed data not available")
            
            # Run pipeline
            df_features = feature_engineering_pipeline(
                processed_path,
                save=False,
                scaling_method='minmax'
            )
            
            # Check output
            assert not df_features.empty
            assert TARGET_COLUMN in df_features.columns
            
            # Check feature count (14 features including target)
            assert len(df_features.columns) == 14
            
            # Check no missing values
            assert df_features.isnull().sum().sum() == 0
            
            # Check scaled features are in [0, 1]
            for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
                if col in df_features.columns:
                    assert df_features[col].min() >= 0.0
                    assert df_features[col].max() <= 1.0
            
            print("✅ Full feature engineering pipeline test passed")
            
        except Exception as e:
            pytest.fail(f"Full pipeline test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])