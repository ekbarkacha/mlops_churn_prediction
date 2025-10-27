import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest
from unittest import mock
from tempfile import TemporaryDirectory

# Import functions from your preprocessing script
from src.data_pipeline.data_ingestion import ingest_from_csv, ingest_from_kaggle
from src.data_pipeline.data_preprocessing import (
    data_cleaning,
    data_encoding,
    validate_schema,
    grab_col_names,
)
from src.data_pipeline.feature_engineering import (
    feature_creation,
    feature_transformation,
    feature_selection,
    feature_scaling,
)

# Example CSV data for testing
TEST_DATA = pd.DataFrame({
    "customerID": ["A_0001", "B_0002"],
    "gender": ["Male", "Female"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["No", "Yes"],
    "tenure": [1, 34],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No phone service", "Yes"],
    "InternetService": ["DSL", "Fiber optic"],
    "OnlineSecurity": ["Yes", "No"],
    "OnlineBackup": ["No", "Yes"],
    "DeviceProtection": ["Yes", "No"],
    "TechSupport": ["No", "Yes"],
    "StreamingTV": ["No", "Yes"],
    "StreamingMovies": ["No", "Yes"],
    "Contract": ["Month-to-month", "One year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Mailed check"],
    "MonthlyCharges": [29.85, 56.95],
    "TotalCharges": ["29.85", "1889.5"],
    "Churn": ["No", "Yes"]
})

EXPECTED_COLUMNS = TEST_DATA.columns.tolist()

# DATA INGEST TEST
# Tests for ingest_from_csv
def test_ingest_from_csv(tmp_path):
    csv_file = tmp_path / "test.csv"
    TEST_DATA.to_csv(csv_file, index=False)
    
    df = ingest_from_csv(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 21)

def test_ingest_from_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        ingest_from_csv("non_existing_file.csv")


# Tests for ingest_from_kaggle
@mock.patch("src.data_pipeline.data_ingestion.KaggleApi")
def test_ingest_from_kaggle(mock_kaggle_api, tmp_path, monkeypatch):
    monkeypatch.setenv("KAGGLE_USERNAME", "dummy")
    monkeypatch.setenv("KAGGLE_KEY", "dummy")

    mock_api_instance = mock_kaggle_api.return_value
    mock_api_instance.authenticate.return_value = None
    mock_api_instance.dataset_download_files.return_value = None

    ingest_from_kaggle("dummy/dataset", str(tmp_path))

    mock_api_instance.authenticate.assert_called_once()
    mock_api_instance.dataset_download_files.assert_called_once_with(
        "dummy/dataset", path=str(tmp_path), unzip=True
    )

# DATA PREPROSSESING TESTS
# Tests for data cleaning
def test_data_cleaning():
    df = TEST_DATA.copy()
    df.loc[0, "TotalCharges"] = "invalid"  # trigger coercion
    df_cleaned = data_cleaning(df)
    assert df_cleaned["TotalCharges"].dtype in [float, int]
    assert df_cleaned["TotalCharges"].iloc[0] == 0

# Tests for data encoding
def test_data_encoding():
    df_encoded = data_encoding(TEST_DATA.copy())
    assert set(df_encoded["Churn"].unique()) == {0, 1}

# Tests for schema validation
def test_validate_schema_pass():
    assert validate_schema(TEST_DATA.copy(), EXPECTED_COLUMNS) is True

def test_validate_schema_fail():
    df_missing = TEST_DATA.copy().drop(columns=["Churn"])
    with pytest.raises(ValueError):
        validate_schema(df_missing, EXPECTED_COLUMNS)

# FEATURE ENGINEERING TESTS
def test_feature_creation():
    df_feat = feature_creation(TEST_DATA.copy())
    # Basic checks
    assert isinstance(df_feat, pd.DataFrame), "feature_creation should return a DataFrame"
    assert df_feat.shape == TEST_DATA.shape, "feature_creation should not change shape by default"

def test_feature_transformation():
    df_feat = feature_transformation(TEST_DATA.copy())
    # Basic checks
    assert isinstance(df_feat, pd.DataFrame), "feature_transformation should return a DataFrame"
    assert df_feat.shape == TEST_DATA.shape, "feature_transformation should not change shape by default"

def test_feature_selection():
    df_feat = feature_selection(TEST_DATA.copy())
    # Columns that should be dropped
    drop_cols = ['customerID','PhoneService','gender','StreamingTV','StreamingMovies','MultipleLines','InternetService']
    for col in drop_cols:
        assert col not in df_feat.columns, f"{col} should be dropped by feature_selection"

def test_feature_scaling():
    # First encode numeric columns for scaling
    df_encoded = data_cleaning(TEST_DATA.copy())
    df_encoded = data_encoding(df_encoded)
    
    df_scaled = feature_scaling(df_encoded.copy())
    # Ensure columns are scaled between 0 and 1
    for col in ['tenure','MonthlyCharges','TotalCharges']:
        assert df_scaled[col].between(0,1).all(), f"{col} should be scaled between 0 and 1"