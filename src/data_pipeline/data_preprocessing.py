"""
Data Preprocessing Module - Clean, validate, and encode data
"""
import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.const import (
    EXPECTED_COLUMNS,
    TARGET_COLUMN,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    EXCLUDE_FROM_ENCODING,
    PREPROCESSORS_DIR,
    LABEL_ENCODERS_FILE,
    PROCESSED_DATA_DIR,
    PROCESSED_FILE_NAME,
    RAW_DATA_DIR,
    RAW_FILE_NAME
)
from src.data_pipeline.data_ingestion import ingest_from_csv, ingest_from_kaggle

# Initialize logger
logger = get_logger(__name__)


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling numeric conversions and missing values.
    
    Steps:
    1. Convert 'TotalCharges' to numeric (some values are empty strings)
    2. Replace invalid or missing values with 0
    
    Args:
        df (pd.DataFrame): Raw dataframe
    
    Returns:
        pd.DataFrame: Cleaned dataframe
    
    Raises:
        RuntimeError: If cleaning fails
    """
    logger.info("🧹 Starting data cleaning...")
    
    try:
        df = df.copy(deep=True)
        
        # Convert TotalCharges to numeric, coercing invalid values to NaN
        logger.info("   • Converting 'TotalCharges' to numeric...")
        original_type = df['TotalCharges'].dtype
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Count NaN values created
        nan_count = df['TotalCharges'].isnull().sum()
        if nan_count > 0:
            logger.warning(f"   ⚠️  Found {nan_count} invalid/missing values in 'TotalCharges'")
        
        # Fill NaN values with 0 (new customers with no charges yet)
        df['TotalCharges'] = df['TotalCharges'].fillna(0)
        logger.info(f"   • Filled {nan_count} missing values with 0")
        
        # Verify no missing values remain
        remaining_nulls = df['TotalCharges'].isnull().sum()
        if remaining_nulls > 0:
            logger.error(f"   ❌ {remaining_nulls} null values still remain!")
            raise ValueError(f"Failed to clean all null values in TotalCharges")
        
        logger.info("✅ Data cleaning completed successfully")
        logger.info(f"   • TotalCharges: {original_type} → {df['TotalCharges'].dtype}")
        logger.info(f"   • Range: [{df['TotalCharges'].min():.2f}, {df['TotalCharges'].max():.2f}]")
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Error during data cleaning: {e}")
        raise RuntimeError(f"Data cleaning failed: {e}") from e


def data_encoding(df: pd.DataFrame, save: bool = False) -> pd.DataFrame:
    """
    Encode categorical columns and target variable ('Churn') numerically.
    
    Steps:
    1. Encode target variable 'Churn' (Yes → 1, No → 0)
    2. Encode categorical features using LabelEncoder
    3. Optionally save encoders for production use
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        save (bool): If True, save encoders to disk
    
    Returns:
        pd.DataFrame: Encoded dataframe
    
    Raises:
        RuntimeError: If encoding fails
    """
    logger.info("🔢 Starting data encoding...")
    
    try:
        df = df.copy(deep=True)
        
        # 1. Encode target variable 'Churn'
        if TARGET_COLUMN in df.columns:
            logger.info(f"   • Encoding target variable '{TARGET_COLUMN}'...")
            original_values = df[TARGET_COLUMN].unique()
            df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'No': 0, 'Yes': 1})
            
            # Verify encoding
            if df[TARGET_COLUMN].isnull().any():
                logger.error(f"   ❌ Target encoding created null values!")
                raise ValueError(f"Target encoding failed - unexpected values: {original_values}")
            
            logger.info(f"   ✅ Target encoded: {list(original_values)} → [0, 1]")
            logger.info(f"   • Class distribution: {df[TARGET_COLUMN].value_counts().to_dict()}")
        
        # 2. Identify categorical columns to encode
        cat_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        # Remove columns to exclude
        for col in EXCLUDE_FROM_ENCODING:
            if col in cat_columns:
                cat_columns.remove(col)
        
        logger.info(f"   • Found {len(cat_columns)} categorical columns to encode")
        logger.info(f"   • Columns: {cat_columns}")
        
        # 3. Encode categorical features
        encoders = {}
        for col in cat_columns:
            logger.info(f"   • Encoding '{col}'...")
            le = LabelEncoder()
            
            # Store original unique values
            original_unique = df[col].unique()
            
            # Fit and transform
            df[col] = le.fit_transform(df[col])
            
            # Store encoder
            encoders[col] = le
            
            logger.info(f"   ✅ '{col}': {len(original_unique)} unique values → [0, {len(original_unique)-1}]")
        
        # 4. Save encoders if requested
        if save:
            logger.info("💾 Saving label encoders...")
            PREPROCESSORS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = PREPROCESSORS_DIR / LABEL_ENCODERS_FILE
            joblib.dump(encoders, save_path)
            logger.info(f"✅ Label encoders saved to: {save_path}")
            logger.info(f"   • Encoders saved: {list(encoders.keys())}")
        
        logger.info("✅ Data encoding completed successfully")
        logger.info(f"   • Total columns encoded: {len(encoders)}")
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Error during data encoding: {e}")
        raise RuntimeError(f"Data encoding failed: {e}") from e


def validate_schema(df: pd.DataFrame, expected_columns: list) -> bool:
    """
    Validate that the dataframe contains all expected columns.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        expected_columns (list): List of expected column names
    
    Returns:
        bool: True if valid
    
    Raises:
        ValueError: If validation fails
    """
    logger.info("🔍 Validating dataframe schema...")
    
    try:
        # Check for missing columns
        missing = [col for col in expected_columns if col not in df.columns]
        
        if missing:
            logger.error(f"❌ Missing columns detected: {missing}")
            raise ValueError(f"Missing columns: {missing}")
        
        # Check for extra columns
        extra = [col for col in df.columns if col not in expected_columns]
        if extra:
            logger.warning(f"⚠️  Extra columns detected (will be ignored): {extra}")
        
        # Check for null values
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning("⚠️  Missing values detected:")
            for col, count in null_counts[null_counts > 0].items():
                logger.warning(f"   • {col}: {count} null values")
        else:
            logger.info("✅ No missing values detected")
        
        logger.info("✅ Schema validation passed successfully")
        return True
    
    except Exception as e:
        logger.error(f"❌ Schema validation failed: {e}")
        raise


def grab_col_names(df: pd.DataFrame, cat_th: int = 10, car_th: int = 20) -> tuple:
    """
    Separate columns into categorical, numerical, and cardinal groups.
    
    Useful for exploratory data analysis.
    
    Args:
        df (pd.DataFrame): The dataframe to analyze
        cat_th (int): Threshold for numeric columns considered categorical
        car_th (int): Threshold for categorical columns considered cardinal
    
    Returns:
        tuple: (cat_cols, num_cols, cat_but_car)
    """
    logger.info("📊 Extracting column type groups...")
    
    try:
        # Identify categorical columns (object type)
        cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
        
        # Numeric but categorical columns (few unique values)
        num_but_cat = [
            col for col in df.columns 
            if df[col].nunique() < cat_th and df[col].dtypes != "O"
        ]
        
        # Categorical but cardinal columns (many unique values)
        cat_but_car = [
            col for col in df.columns 
            if df[col].nunique() > car_th and df[col].dtypes == "O"
        ]
        
        # Combine categorical columns
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]
        
        # Identify numerical columns
        num_cols = [col for col in df.columns if df[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]
        
        logger.info(f"✅ Column extraction complete:")
        logger.info(f"   • Categorical columns: {len(cat_cols)}")
        logger.info(f"   • Numerical columns: {len(num_cols)}")
        logger.info(f"   • Cardinal columns: {len(cat_but_car)}")
        
        return cat_cols, num_cols, cat_but_car
    
    except Exception as e:
        logger.error(f"❌ Error during column name extraction: {e}")
        raise RuntimeError(f"Column extraction failed: {e}") from e


def data_processing_pipeline(save: bool = False, force_download: bool = False) -> Path:
    """
    Complete data preprocessing pipeline.
    
    Steps:
    1. Check if raw data exists, download from Kaggle if not
    2. Load raw data
    3. Validate schema
    4. Clean data (TotalCharges conversion)
    5. Encode categorical features
    6. Save processed data
    
    Args:
        save (bool): If True, save encoders to disk
        force_download (bool): If True, force re-download from Kaggle
    
    Returns:
        Path: Path to processed data file
    
    Raises:
        RuntimeError: If pipeline fails
    """
    logger.info("=" * 80)
    logger.info("🚀 Starting Data Preprocessing Pipeline")
    logger.info("=" * 80)
    
    try:
        # 1. Check and load raw data
        raw_data_path = RAW_DATA_DIR / RAW_FILE_NAME
        
        if not raw_data_path.exists() or force_download:
            logger.info("📥 Raw data not found locally, downloading from Kaggle...")
            ingest_from_kaggle()
        
        logger.info(f"📂 Loading raw data from: {raw_data_path}")
        df = ingest_from_csv(str(raw_data_path))
        logger.info(f"✅ Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # 2. Validate schema
        validate_schema(df, EXPECTED_COLUMNS)
        
        # 3. Clean the data
        df = data_cleaning(df)
        
        # 4. Encode categoricals
        df = data_encoding(df, save=save)
        
        # 5. Save processed data
        processed_path = PROCESSED_DATA_DIR / PROCESSED_FILE_NAME
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving processed data to: {processed_path}")
        df.to_csv(processed_path, index=False)
        logger.info(f"✅ Processed data saved ({df.shape[0]} rows, {df.shape[1]} columns)")
        
        # 6. Summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("📊 PREPROCESSING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"   • Input shape: {df.shape}")
        logger.info(f"   • Columns: {list(df.columns)}")
        logger.info(f"   • Data types:")
        for dtype in df.dtypes.unique():
            cols = df.select_dtypes(include=[dtype]).columns.tolist()
            logger.info(f"     - {dtype}: {len(cols)} columns")
        logger.info(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"   • Missing values: {df.isnull().sum().sum()}")
        logger.info("=" * 80)
        
        logger.info("✅ Data preprocessing pipeline completed successfully")
        
        return processed_path
    
    except Exception as e:
        logger.error(f"❌ Data preprocessing pipeline failed: {e}")
        raise RuntimeError(f"Preprocessing pipeline failed: {e}") from e


def main():
    """Main function for testing preprocessing module"""
    try:
        # Run preprocessing pipeline
        processed_path = data_processing_pipeline(save=True, force_download=False)
        
        # Load and display processed data
        df_processed = pd.read_csv(processed_path)
        
        print("\n" + "=" * 80)
        print("📊 PROCESSED DATA PREVIEW")
        print("=" * 80)
        print(df_processed.head(10))
        
        print("\n" + "=" * 80)
        print("📈 DATA TYPES")
        print("=" * 80)
        print(df_processed.dtypes)
        
        print("\n" + "=" * 80)
        print("📊 DESCRIPTIVE STATISTICS")
        print("=" * 80)
        print(df_processed.describe())
        
        print("\n" + "=" * 80)
        print("🎯 TARGET DISTRIBUTION")
        print("=" * 80)
        print(df_processed['Churn'].value_counts())
        print(f"\nChurn rate: {df_processed['Churn'].mean()*100:.2f}%")
        
    except Exception as e:
        logger.error(f"❌ Preprocessing test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()