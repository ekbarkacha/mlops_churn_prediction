"""
Data Ingestion Module - Download and load data from Kaggle or local CSV
"""
import os
import sys
from pathlib import Path
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.config import (
    KAGGLE_USERNAME, 
    KAGGLE_KEY, 
    KAGGLE_DATASET,
    RAW_DATA_DIR
)
from src.utils.const import RAW_FILE_NAME, EXPECTED_COLUMNS

# Configure Kaggle credentials
os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY


# Initialize logger
logger = get_logger(__name__)


def ingest_from_csv(path: str) -> pd.DataFrame:
    """
    Load data from a local CSV file.
    
    Args:
        path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataframe
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or unparseable
    """
    logger.info(f"📂 Loading data from CSV: {path}")
    
    # Check file exists
    if not os.path.exists(path):
        logger.error(f"❌ File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        # Load CSV
        df = pd.read_csv(path)
        
        # Validate not empty
        if df.empty:
            logger.error(f"❌ The CSV file '{path}' is empty")
            raise ValueError(f"The CSV file '{path}' is empty")
        
        logger.info(f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    except pd.errors.EmptyDataError:
        logger.error(f"❌ The CSV file '{path}' is empty")
        raise ValueError(f"The CSV file '{path}' is empty")
    
    except pd.errors.ParserError:
        logger.error(f"❌ The CSV file '{path}' could not be parsed")
        raise ValueError(f"The CSV file '{path}' could not be parsed")
    
    except Exception as e:
        logger.exception(f"❌ Unexpected error reading '{path}': {e}")
        raise RuntimeError(f"Unexpected error reading '{path}': {e}")


def ingest_from_kaggle(
    dataset_name: str = KAGGLE_DATASET,
    download_path: Path = RAW_DATA_DIR
) -> Path:
    """
    Download and extract a Kaggle dataset.
    
    Args:
        dataset_name (str): Kaggle dataset identifier (e.g., 'blastchar/telco-customer-churn')
        download_path (Path): Directory where data will be saved
    
    Returns:
        Path: Path to the downloaded CSV file
    
    Raises:
        RuntimeError: If download fails
    """
    logger.info(f"🌐 Downloading dataset from Kaggle: {dataset_name}")
    
    # Create directory if doesn't exist
    download_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Authenticate Kaggle API
        api = KaggleApi()
        api.authenticate()
        logger.info("✅ Kaggle API authenticated successfully")
        
        # Download dataset
        logger.info(f"⬇️ Downloading to: {download_path}")
        api.dataset_download_files(
            dataset_name,
            path=str(download_path),
            unzip=True
        )
        
        logger.info(f"✅ Dataset downloaded and extracted to '{download_path}'")
        
        # Return path to CSV file
        csv_path = download_path / RAW_FILE_NAME
        
        if not csv_path.exists():
            logger.error(f"❌ Expected file not found: {csv_path}")
            raise FileNotFoundError(f"Expected file not found: {csv_path}")
        
        logger.info(f"✅ CSV file located at: {csv_path}")
        return csv_path
    
    except FileNotFoundError as e:
        logger.error(f"❌ The download path '{download_path}' could not be created or found")
        raise FileNotFoundError(f"The download path '{download_path}' could not be created or found") from e
    
    except Exception as e:
        logger.exception(f"❌ Failed to download dataset '{dataset_name}': {e}")
        raise RuntimeError(f"Failed to download dataset '{dataset_name}': {e}") from e


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the ingested dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
    
    Returns:
        bool: True if valid, raises exception otherwise
    
    Raises:
        ValueError: If validation fails
    """
    logger.info("🔍 Validating ingested data...")
    
    # Check not empty
    if df.empty:
        logger.error("❌ Dataframe is empty")
        raise ValueError("Dataframe is empty")
    
    # Check columns
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        logger.error(f"❌ Missing columns: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Check data types
    logger.info(f"📊 Data types:\n{df.dtypes}")
    
    # Check missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        logger.warning(f"⚠️ Missing values detected:\n{missing_counts[missing_counts > 0]}")
    else:
        logger.info("✅ No missing values detected")
    
    # Check duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"⚠️ {duplicates} duplicate rows detected")
    else:
        logger.info("✅ No duplicate rows detected")
    
    # Summary stats
    logger.info(f"📈 Dataset summary:")
    logger.info(f"   • Rows: {df.shape[0]}")
    logger.info(f"   • Columns: {df.shape[1]}")
    logger.info(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    logger.info("✅ Data validation passed")
    return True


def ingest_data(force_download: bool = False) -> pd.DataFrame:
    """
    Main ingestion function - orchestrates the data ingestion process.
    
    Args:
        force_download (bool): If True, force download from Kaggle even if local file exists
    
    Returns:
        pd.DataFrame: Ingested and validated dataframe
    """
    logger.info("=" * 80)
    logger.info("🚀 Starting Data Ingestion Process")
    logger.info("=" * 80)
    
    csv_path = RAW_DATA_DIR / RAW_FILE_NAME
    
    # Check if file exists locally
    if csv_path.exists() and not force_download:
        logger.info(f"📁 Local file found: {csv_path}")
        logger.info("💡 Loading from local file (use force_download=True to re-download)")
        df = ingest_from_csv(str(csv_path))
    else:
        logger.info("🌐 Local file not found or force download requested")
        logger.info("⬇️ Downloading from Kaggle...")
        downloaded_path = ingest_from_kaggle()
        df = ingest_from_csv(str(downloaded_path))
    
    # Validate data
    validate_data(df)
    
    logger.info("=" * 80)
    logger.info("✅ Data Ingestion Process Completed Successfully")
    logger.info("=" * 80)
    
    return df


# For testing
if __name__ == "__main__":
    # Test ingestion
    try:
        data = ingest_data(force_download=False)
        print("\n" + "=" * 80)
        print("📊 DATASET PREVIEW")
        print("=" * 80)
        print(data.head())
        print("\n" + "=" * 80)
        print("📈 DATASET INFO")
        print("=" * 80)
        print(data.info())
    except Exception as e:
        logger.error(f"❌ Ingestion test failed: {e}")
        sys.exit(1)