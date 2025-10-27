import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils.logger import get_logger
from src.data_pipeline.data_ingestion import ingest_from_kaggle,ingest_from_csv
from src.utils.const import KAGGLE_DATASET,EXPECTED_COLUMNS,RAW_DATA_DIR,PROCESSED_DATA_DIR,raw_file_name,processed_file_name

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

def data_encoding(df: pd.DataFrame) -> pd.DataFrame:
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
        le = LabelEncoder()
        df = df.copy(deep=True)
        text_data_features = [col for col in df.columns if col not in df.describe().columns and col != "customerID"]

        # Encode each categorical column
        for col in text_data_features:
            df[col] = le.fit_transform(df[col])

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

def main():
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
    df = data_encoding(df)

    # Save processed data
    PROCESSED_PATH = f"{PROCESSED_DATA_DIR}/{processed_file_name}"
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    logger.info(f"Processed data saved to {PROCESSED_PATH} ({df.shape[0]} rows)")

    logger.info("Data preprocessing completed..")

if __name__ == "__main__":
    main()


