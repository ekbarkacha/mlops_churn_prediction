"""
Module: data_preprocessing.py
==============================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description: 
------------
This module provides a complete data preprocessing pipeline for a dataset, including:
- Data ingestion (from local CSV or Kaggle) imported from data_ingestion.py
- Data cleaning (handling missing and invalid values)
- Data encoding (label encoding categorical variables)
- Schema validation
- Column categorization (categorical, numerical, cardinal)
- Saving processed data and optional DVC tracking

"""
# Imports and setup
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import pandas as pd
import threading
from sklearn.preprocessing import LabelEncoder

# Custom utility imports
from src.utils.logger import get_logger
from src.data_pipeline.data_versioning_dvc import dvc_track_processed_file
from src.data_pipeline.data_ingestion import ingest_from_kaggle,ingest_from_csv
from src.utils.const import (KAGGLE_DATASET,EXPECTED_COLUMNS,RAW_DATA_DIR,
                             PROCESSED_DATA_DIR,raw_file_name,processed_file_name,
                             PREPROCESSORS,label_encoders_file_name)

logger = get_logger(__name__)

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling numeric conversions and missing values.

    - Converts 'TotalCharges' to numeric.
    - Replaces invalid or missing values with 0.
    """
    logger.info("Starting data cleaning...")

    try:
        # Convert TotalCharges to numeric, coercing invalid values to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Fill NaN values with 0
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

        logger.info("Data cleaning completed successfully.")
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise RuntimeError(f"Data cleaning failed: {e}")

    # df.drop('customerID', axis=1, inplace=True)

    return df

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling numeric conversions and missing values.

    Steps:
    - Converts 'TotalCharges' to numeric.
    - Ensures 'SeniorCitizen' is categorical.
    - Handles missing values:
        * Numeric columns → fill with median.
        * Categorical columns → fill with mode.
    """
    logger.info("Starting data cleaning...")

    try:
        # Convert TotalCharges to numeric, coercing invalid values to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Fill NaN values with 0
        df['TotalCharges'] = df['TotalCharges'].fillna(0)

        # Ensure 'SeniorCitizen' is treated as categorical
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype('category')

        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(exclude=['number']).columns

        # Fill missing values for numeric columns with median
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.debug(f"Filled NaNs in numeric column '{col}' with median: {median_value}")

        # Fill missing values for categorical columns with mode
        for col in categorical_cols:
            if df[col].isna().sum() > 0:
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
                logger.debug(f"Filled NaNs in categorical column '{col}' with mode: {mode_value}")

        logger.info("Data cleaning completed successfully.")

    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise RuntimeError(f"Data cleaning failed: {e}")

    return df

def data_encoding(df: pd.DataFrame,save=False) -> pd.DataFrame:
    """
    Encode categorical columns and target variable ('Churn') numerically.

    - Maps 'Churn' to 0 (No) and 1 (Yes).
    - Encodes all text columns using LabelEncoder.
    """
    logger.info("Starting data encoding...")

    try:
        # Encode target variable 'Churn'
        if "Churn" in df.columns:
            df["Churn"] = df["Churn"].map({'No': 0, 'Yes': 1})

        # Identify categorical columns (non-numeric)
        df = df.copy(deep=True)
        cat_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if "customerID" in cat_columns:
            cat_columns.remove("customerID")

        encoders = {}
        for col in cat_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        if save:    
            os.makedirs(PREPROCESSORS,exist_ok=True)
            save_path = os.path.join(PREPROCESSORS, label_encoders_file_name)
            joblib.dump(encoders, save_path)
            logger.info(f"Label encoder saved to {save_path}")

        logger.info("Data encoding completed successfully.")
    except Exception as e:
        logger.error(f"Error during data encoding: {e}")
        raise RuntimeError(f"Data encoding failed: {e}")

    return df


def validate_schema(df: pd.DataFrame, expected_columns: list):
    """
    Validate that the dataframe contains all expected columns.

    Raises:
        ValueError: If any expected columns are missing.
    """
    logger.info("Validating dataframe schema...")

    try:
        # Identify missing columns
        missing = [col for col in expected_columns if col not in df.columns]
        if df.isnull().sum().any():
            logger.warning("Missing values detected in dataframe.")

        if missing:
            logger.error(f"Missing columns detected: {missing}")
            raise ValueError(f"Missing columns: {missing}")

        logger.info("Schema validation passed successfully.")
        return True

    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        raise


def grab_col_names(df: pd.DataFrame, cat_th=10, car_th=20):
    """
    Separate columns into categorical, numerical, and cardinal groups.

    Args:
        df (pd.DataFrame): The dataframe to analyze.
        cat_th (int): Threshold for numeric columns considered categorical.
        car_th (int): Threshold for categorical columns considered cardinal.

    Returns:
        tuple: (cat_cols, num_cols, cat_but_car)
    """
    logger.info("Extracting column type groups...")

    try:
        # Identify categorical columns
        cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

        # Numeric but categorical columns
        num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes != "O"]

        # Categorical but cardinal columns
        cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and df[col].dtypes == "O"]

        # Combine categorical columns
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # Identify numerical columns
        num_cols = [col for col in df.columns if df[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        logger.info(f"Column extraction complete. cat_cols={len(cat_cols)}, num_cols={len(num_cols)}, cat_but_car={len(cat_but_car)}")

        return cat_cols, num_cols, cat_but_car

    except Exception as e:
        logger.error(f"Error during column name extraction: {e}")
        raise RuntimeError(f"Column extraction failed: {e}")
    
def data_processing_pipeline(save=False):
    """
    Run the complete data preprocessing pipeline:
    - Check and ingest raw data
    - Validate schema
    - Clean data
    - Encode categorical variables
    - Save processed data
    - Optionally track processed file using DVC

    Args:
        save (bool, optional): Whether to save encoders and track processed file. Defaults to False.

    Returns:
        str: Path to the processed CSV file.
    """
    logger.info("Starting data preprocessing...")
    RAW_DATA_PATH = f"{RAW_DATA_DIR}/{raw_file_name}"
    
    # Check Raw data if not available download from kaggle
    if not os.path.exists(RAW_DATA_PATH):
        ingest_from_kaggle(KAGGLE_DATASET,RAW_DATA_DIR)    
        RAW_DATA_PATH = f"{RAW_DATA_DIR}/{raw_file_name}"

    # Ingest from csv
    df = ingest_from_csv(RAW_DATA_PATH)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    #Validate schema
    validate_schema(df, EXPECTED_COLUMNS)

    # Clean the data
    df = data_cleaning(df)

    # Encode categoricals
    df = data_encoding(df,save)

    # Save processed data
    PROCESSED_PATH = f"{PROCESSED_DATA_DIR}/{processed_file_name}"
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"Processed data saved to {PROCESSED_PATH} ({df.shape[0]} rows)")

    logger.info("Data preprocessing completed..")

    if save:
        # Run DVC tracking in a background thread
        threading.Thread(target=dvc_track_processed_file, args=(logger, RAW_DATA_PATH), daemon=True).start()


    return PROCESSED_PATH

def main():
    processed_path = data_processing_pipeline()

if __name__ == "__main__":
    main()


