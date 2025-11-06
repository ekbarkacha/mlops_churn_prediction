"""
Module: test_data_pipeline.py — Unit Tests for Data Pipeline Components
====================================================================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description:
------------

This module contains **unit tests** for all stages of the MLOps data pipeline, 
including ingestion, preprocessing and feature engineering.

Purpose:
--------
To ensure each component in the data pipeline behaves as expected — verifying 
data transformations, integrity, and schema consistency before the modeling phase.

Covers:
--------
1. Data Ingestion
   - Reading datasets from local CSVs and Kaggle API.

2. Data Preprocessing
   - Cleaning, encoding and schema validation.

3. Feature Engineering
   - Feature creation, transformation, selection and scaling.

"""
# Imports and setup
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest
from unittest import mock

# Custom utility imports
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

### Sample Test Dataset Setup ##
# -------------------------------
# This dataset simulates a small customer churn dataset for testing.
# It includes categorical and numeric columns along with a binary target 'Churn'.
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

# Expected schema for testing validation
EXPECTED_COLUMNS = TEST_DATA.columns.tolist()


## DATA INGESTION TESTS

def test_ingest_from_csv(tmp_path):
    """
    Test ingestion of data from a local CSV file.

    Ensures:
    - The function reads a CSV correctly.
    - Returns a valid pandas DataFrame with expected shape.
    """
    csv_file = tmp_path / "test.csv"
    TEST_DATA.to_csv(csv_file, index=False)

    df = ingest_from_csv(str(csv_file))

    assert isinstance(df, pd.DataFrame), "Output should be a pandas DataFrame"
    assert df.shape == (2, 21), "Shape mismatch — expected 2 rows and 21 columns"


def test_ingest_from_csv_file_not_found():
    """
    Test that ingest_from_csv raises FileNotFoundError for invalid paths.
    """
    with pytest.raises(FileNotFoundError):
        ingest_from_csv("non_existing_file.csv")

@mock.patch("src.data_pipeline.data_ingestion.KaggleApi")
def test_ingest_from_kaggle(mock_kaggle_api, tmp_path, monkeypatch):
    """
    Test ingestion from Kaggle datasets using a mocked Kaggle API.

    Ensures:
    - Kaggle API is authenticated and download is invoked correctly.
    """
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


## DATA PREPROCESSING TESTS

def test_data_cleaning():
    """
    Test the data_cleaning function.

    Scenario:
    - Introduces an invalid 'TotalCharges' value to test numeric coercion.
    - Ensures invalid entries are converted to 0 or handled gracefully.
    """
    df = TEST_DATA.copy()
    df.loc[0, "TotalCharges"] = "invalid"  # trigger coercion to 0
    df_cleaned = data_cleaning(df)

    assert df_cleaned["TotalCharges"].dtype in [float, int], "TotalCharges should be numeric"
    assert df_cleaned["TotalCharges"].iloc[0] == 0, "Invalid TotalCharges should be coerced to 0"


def test_data_encoding():
    """
    Test the data_encoding function.

    Ensures:
    - The target column 'Churn' is encoded as binary (0 and 1).
    """
    df_encoded = data_encoding(TEST_DATA.copy())
    assert set(df_encoded["Churn"].unique()) == {0, 1}, "Churn column not properly encoded"

def test_validate_schema_pass():
    """
    Test validate_schema with correct schema.
    """
    assert validate_schema(TEST_DATA.copy(), EXPECTED_COLUMNS) is True


def test_validate_schema_fail():
    """
    Test validate_schema with missing column to ensure it raises ValueError.
    """
    df_missing = TEST_DATA.copy().drop(columns=["Churn"])
    with pytest.raises(ValueError):
        validate_schema(df_missing, EXPECTED_COLUMNS)


## FEATURE ENGINEERING TESTS

def test_feature_creation():
    """
    Test feature_creation function.

    Ensures:
    - Returns a DataFrame.
    - Does not alter dataset shape unless explicitly designed to.
    """
    df_feat = feature_creation(TEST_DATA.copy())
    assert isinstance(df_feat, pd.DataFrame), "feature_creation should return a DataFrame"
    assert df_feat.shape == TEST_DATA.shape, "feature_creation should not change shape by default"


def test_feature_transformation():
    """
    Test feature_transformation function.

    Ensures:
    - Transformation does not break data structure.
    """
    df_feat = feature_transformation(TEST_DATA.copy())
    assert isinstance(df_feat, pd.DataFrame), "feature_transformation should return a DataFrame"
    assert df_feat.shape == TEST_DATA.shape, "feature_transformation should not change shape by default"


def test_feature_selection():
    """
    Test feature_selection function.

    Ensures:
    - Redundant or ID-like columns are dropped as expected.
    """
    df_feat = feature_selection(TEST_DATA.copy())
    drop_cols = ['customerID', 'PhoneService', 'gender', 'StreamingTV', 
                 'StreamingMovies', 'MultipleLines', 'InternetService']

    for col in drop_cols:
        assert col not in df_feat.columns, f"{col} should be dropped by feature_selection"


def test_feature_scaling():
    """
    Test feature_scaling function.

    Ensures:
    - Numeric features are scaled between 0 and 1.
    - Scaling preserves data shape and type consistency.
    """
    df_encoded = data_cleaning(TEST_DATA.copy())
    df_encoded = data_encoding(df_encoded)

    df_scaled = feature_scaling(df_encoded.copy())

    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        assert df_scaled[col].between(0, 1).all(), f"{col} should be scaled between 0 and 1"
