"""
Unit tests for data preprocessing module
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.data_preprocessing import (
    data_cleaning,
    data_encoding,
    validate_schema,
    grab_col_names,
    data_processing_pipeline
)
from src.utils.const import EXPECTED_COLUMNS, TARGET_COLUMN


class TestDataCleaning:
    """Test suite for data cleaning"""
    
    def test_totalcharges_conversion(self):
        """Test TotalCharges conversion to numeric"""
        # Create mock data with string TotalCharges
        df = pd.DataFrame({
            'TotalCharges': ['100.5', '200.3', ' ', '300.0', ''],
            'tenure': [1, 2, 3, 4, 5]
        })
        
        df_cleaned = data_cleaning(df)
        
        # Check conversion
        assert df_cleaned['TotalCharges'].dtype in [np.float64, np.int64]
        
        # Check NaN replacement with 0
        assert df_cleaned['TotalCharges'].isnull().sum() == 0
        assert (df_cleaned['TotalCharges'] >= 0).all()
    
    def test_cleaning_preserves_valid_data(self):
        """Test that cleaning doesn't alter valid numeric data"""
        df = pd.DataFrame({
            'TotalCharges': [100.5, 200.3, 300.0],
            'tenure': [1, 2, 3]
        })
        
        df_cleaned = data_cleaning(df)
        
        # Values should remain the same
        assert df_cleaned['TotalCharges'].tolist() == [100.5, 200.3, 300.0]


class TestDataEncoding:
    """Test suite for data encoding"""
    
    def test_target_encoding(self):
        """Test Churn target encoding"""
        df = pd.DataFrame({
            'Churn': ['Yes', 'No', 'Yes', 'No', 'Yes'],
            'customerID': ['C1', 'C2', 'C3', 'C4', 'C5']
        })
        
        df_encoded = data_encoding(df, save=False)
        
        # Check encoding
        assert df_encoded['Churn'].dtype in [np.int64, np.int32]
        assert set(df_encoded['Churn'].unique()) == {0, 1}
        assert (df_encoded['Churn'] == [1, 0, 1, 0, 1]).all()
    
    def test_categorical_encoding(self):
        """Test categorical feature encoding"""
        df = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male', 'Female'],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
            'Churn': ['Yes', 'No', 'Yes', 'No'],
            'customerID': ['C1', 'C2', 'C3', 'C4']
        })
        
        df_encoded = data_encoding(df, save=False)
        
        # Check that categorical columns are now numeric
        assert df_encoded['gender'].dtype in [np.int64, np.int32]
        assert df_encoded['Contract'].dtype in [np.int64, np.int32]
        
        # Check customerID is not encoded
        assert df_encoded['customerID'].dtype == object
    
    def test_encoding_consistency(self):
        """Test that same values get same encoding"""
        df = pd.DataFrame({
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Churn': ['No', 'No', 'No', 'No', 'No'],
            'customerID': ['C1', 'C2', 'C3', 'C4', 'C5']
        })
        
        df_encoded = data_encoding(df, save=False)
        
        # All 'Male' should have same encoding
        male_encodings = df_encoded.loc[df['gender'] == 'Male', 'gender'].unique()
        assert len(male_encodings) == 1


class TestSchemaValidation:
    """Test suite for schema validation"""
    
    def test_valid_schema(self):
        """Test validation passes for valid schema"""
        df = pd.DataFrame({col: [1, 2, 3] for col in EXPECTED_COLUMNS})
        assert validate_schema(df, EXPECTED_COLUMNS) == True
    
    def test_missing_columns(self):
        """Test validation fails for missing columns"""
        df = pd.DataFrame({
            'customerID': [1, 2, 3],
            'gender': ['M', 'F', 'M']
        })
        
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(df, EXPECTED_COLUMNS)


class TestColumnGrouping:
    """Test suite for column grouping"""
    
    def test_grab_col_names(self):
        """Test column type separation"""
        df = pd.DataFrame({
            'cat1': ['A', 'B', 'C'],
            'cat2': ['X', 'Y', 'Z'],
            'num1': [1.5, 2.5, 3.5],
            'num2': [10, 20, 30],
            'binary': [0, 1, 0]
        })
        
        cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)
        
        # Check types
        assert isinstance(cat_cols, list)
        assert isinstance(num_cols, list)
        assert isinstance(cat_but_car, list)
        
        # Check categorical columns identified
        assert 'cat1' in cat_cols
        assert 'cat2' in cat_cols


class TestPreprocessingPipeline:
    """Integration test for full preprocessing pipeline"""
    
    def test_full_pipeline(self):
        """Test complete preprocessing pipeline"""
        try:
            processed_path = data_processing_pipeline(save=True, force_download=False)
            
            # Check file exists
            assert processed_path.exists()
            
            # Load processed data
            df = pd.read_csv(processed_path)
            
            # Check not empty
            assert not df.empty
            
            # Check target is encoded
            assert df['Churn'].dtype in [np.int64, np.int32]
            assert set(df['Churn'].unique()).issubset({0, 1})
            
            # Check no missing values
            assert df.isnull().sum().sum() == 0
            
            print("✅ Full preprocessing pipeline test passed")
            
        except Exception as e:
            pytest.fail(f"Full preprocessing pipeline test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])