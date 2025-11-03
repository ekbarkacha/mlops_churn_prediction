import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger
from src.data_pipeline.data_ingestion import ingest_from_kaggle, ingest_from_csv
from src.utils.const import (
    KAGGLE_DATASET, EXPECTED_COLUMNS, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    raw_file_name, processed_file_name, PREPROCESSORS, label_encoders_file_name
)

logger = get_logger(__name__)

# ===============================================================
#  Step 1: Data Cleaning                                        #
# ===============================================================
def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning:
      - Convert TotalCharges to numeric
      - Smart imputation: MonthlyCharges * tenure for NaN
      - Remove spaces and duplicates
      - Fix SeniorCitizen type
    """
    logger.info("Starting data cleaning...")

    try:
        # Convert TotalCharges to numeric (with coercion)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

        # Smart imputation (MonthlyCharges × tenure)
        missing_total = df['TotalCharges'].isna().sum()
        if missing_total > 0:
            logger.warning(f"Detected {missing_total} missing values in TotalCharges. Imputing...")
            df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'], inplace=True)

        # Fix SeniorCitizen type
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].astype(int).astype(str)
            df['SeniorCitizen'] = df['SeniorCitizen'].replace({'0': 'No', '1': 'Yes'})

        # Remove customer ID column
        if 'customerID' in df.columns:
            df.drop(columns=['customerID'], inplace=True)
            logger.info("Column 'customerID' removed successfully.")

        logger.info("Data cleaning completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise RuntimeError(f"Data cleaning failed: {e}")


# ===============================================================
#  Step 2: Encoding Categorical Variables                       #
# ===============================================================
def data_encoding(df: pd.DataFrame, save=False) -> pd.DataFrame:
    """
    Encoding categorical variables:
      - 'Churn': Yes/No → 1/0
      - Other categorical variables: LabelEncoder
      - Optional saving of encoders
    """
    logger.info("Starting data encoding...")

    try:
        df = df.copy(deep=True)

        # Encode target variable
        if 'Churn' in df.columns:
            df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

        # Identify categorical columns
        cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

        encoders = {}
        for col in cat_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        if save:
            os.makedirs(PREPROCESSORS, exist_ok=True)
            save_path = os.path.join(PREPROCESSORS, label_encoders_file_name)
            joblib.dump(encoders, save_path)
            logger.info(f"Label encoders saved to {save_path}")

        logger.info("Data encoding completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during data encoding: {e}")
        raise RuntimeError(f"Data encoding failed: {e}")


# ===============================================================
#  Step 3: Schema Validation                                    #
# ===============================================================
def validate_schema(df: pd.DataFrame, expected_columns: list):
    """
    Checks that the DataFrame contains all expected columns.
    """
    logger.info("Validating dataframe schema...")
    try:
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


# ===============================================================
#  Step 4: Extract Column Names                                 #
# ===============================================================
def grab_col_names(df: pd.DataFrame, cat_th=10, car_th=20):
    """
    Separates columns into categorical, numerical, and cardinal.
    """
    logger.info("Extracting column type groups...")
    try:
        cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
        num_but_cat = [col for col in df.columns if df[col].nunique() < cat_th and df[col].dtypes != "O"]
        cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and df[col].dtypes == "O"]

        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        num_cols = [col for col in df.columns if df[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        logger.info(f"Column extraction complete. cat_cols={len(cat_cols)}, num_cols={len(num_cols)}, cat_but_car={len(cat_but_car)}")
        return cat_cols, num_cols, cat_but_car
    except Exception as e:
        logger.error(f"Error during column name extraction: {e}")
        raise RuntimeError(f"Column extraction failed: {e}")


# ===============================================================
#  Step 5: Full Preprocessing Pipeline                          #
# ===============================================================
def data_processing_pipeline(save=False):
    """
    Runs the entire preprocessing pipeline:
      - Download
      - Schema validation
      - Cleaning
      - Encoding
      - Final saving
    """
    logger.info("Starting data preprocessing...")

    RAW_DATA_PATH = f"{RAW_DATA_DIR}/{raw_file_name}"

    if not os.path.exists(RAW_DATA_PATH):
        ingest_from_kaggle(KAGGLE_DATASET, RAW_DATA_DIR)

    df = ingest_from_csv(RAW_DATA_PATH)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    validate_schema(df, EXPECTED_COLUMNS)
    df = data_cleaning(df)
    df = data_encoding(df, save)

    PROCESSED_PATH = f"{PROCESSED_DATA_DIR}/{processed_file_name}"
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"Processed data saved to {PROCESSED_PATH} ({df.shape[0]} rows)")

    logger.info("Data preprocessing completed successfully.")
    return PROCESSED_PATH


# ===============================================================
#  Main Entry Point                                             #
# ===============================================================
def main():
    processed_path = data_processing_pipeline(save=True)
    print(f"✅ Data processing done. Processed file available at: {processed_path}")

if __name__ == "__main__":
    main()