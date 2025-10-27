import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from src.utils.logger import get_logger
from src.utils.const import PROCESSED_DATA_DIR,processed_file_name,feature_file_name

logger = get_logger(__name__)

def feature_creation(df: pd.DataFrame) -> pd.DataFrame:
    return df

def feature_transformation(df: pd.DataFrame) -> pd.DataFrame:
    return df

def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    drop_features = ['customerID','PhoneService', 'gender', 'StreamingTV', 'StreamingMovies', 'MultipleLines', 'InternetService']
    df.drop(columns=drop_features, inplace=True, errors='ignore')

    return df

def feature_scaling(df: pd.DataFrame, method='standard') -> pd.DataFrame:
    mms = MinMaxScaler()
    # for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
    #     df[col] = mms.fit_transform(df[[col]])
    # return df
    for col in df.columns:
        df[col] = mms.fit_transform(df[[col]])
    return df

def main():
    logger.info("Starting feature engineering..")

    PROCESSED_DATA = f"{PROCESSED_DATA_DIR}/{processed_file_name}"

    if not os.path.exists(PROCESSED_DATA):
        logger.error(f"Processed data not found at {PROCESSED_DATA}")
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA}")

    df = pd.read_csv(PROCESSED_DATA)
    logger.info(f"Loaded preprocessed data: {df.shape}")

    # Apply feature engineering
    df = feature_creation(df)
    df = feature_transformation(df)
    df = feature_selection(df)
    df = feature_scaling(df)    

    # Save features
    FEATURE_DATA = f"{PROCESSED_DATA_DIR}/{feature_file_name}"
    os.makedirs(os.path.dirname(FEATURE_DATA), exist_ok=True)
    df.to_csv(FEATURE_DATA, index=False)
    logger.info(f"Feature data saved to {FEATURE_DATA} ({df.shape[0]} rows, {df.shape[1]} cols)")

    logger.info("Feature engineering complete")

if __name__ == "__main__":
    main()

