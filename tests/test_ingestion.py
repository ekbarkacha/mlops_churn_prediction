"""
Unit tests for data ingestion module
"""
import pytest
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.data_ingestion import (
    ingest_from_csv,
    validate_data,
    ingest_data
)
from src.utils.const import EXPECTED_COLUMNS


class TestDataIngestion:
    """Test suite for data ingestion"""
    
    def test_ingest_from_csv_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent file"""
        with pytest.raises(FileNotFoundError):
            ingest_from_csv("non_existent_file.csv")
    
    def test_validate_data_empty_dataframe(self):
        """Test validation fails for empty dataframe"""
        df_empty = pd.DataFrame()
        with pytest.raises(ValueError, match="Dataframe is empty"):
            validate_data(df_empty)
    
    def test_validate_data_missing_columns(self):
        """Test validation fails for missing columns"""
        df_incomplete = pd.DataFrame({
            'customerID': [1, 2, 3],
            'gender': ['M', 'F', 'M']
        })
        with pytest.raises(ValueError, match="Missing columns"):
            validate_data(df_incomplete)
    
    def test_validate_data_complete(self):
        """Test validation passes for complete dataframe"""
        # Create mock dataframe with all expected columns
        df_complete = pd.DataFrame({col: [1, 2, 3] for col in EXPECTED_COLUMNS})
        assert validate_data(df_complete) == True


def test_full_ingestion():
    """Integration test - full ingestion process"""
    try:
        df = ingest_data(force_download=False)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in EXPECTED_COLUMNS)
        print("✅ Full ingestion test passed")
    except Exception as e:
        pytest.fail(f"Full ingestion test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])