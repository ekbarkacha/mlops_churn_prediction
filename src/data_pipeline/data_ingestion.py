import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
from src.utils.logger import get_logger
from src.utils.config import KAGGLE_USERNAME,KAGGLE_KEY

os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY
    
from kaggle.api.kaggle_api_extended import KaggleApi

logger = get_logger(__name__)

def ingest_from_csv(path: str) -> pd.DataFrame:
    """Load data from a local CSV file."""
    logger.info(f"Loading data from CSV: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        logger.error(f"The CSV file '{path}' is empty.")
        raise ValueError(f"The CSV file '{path}' is empty.")
    except pd.errors.ParserError:
        logger.error(f"The CSV file '{path}' could not be parsed.")
        raise ValueError(f"The CSV file '{path}' could not be parsed.")
    except Exception as e:
        logger.exception(f"Unexpected error reading '{path}': {e}")
        raise RuntimeError(f"Unexpected error reading '{path}': {e}")

def ingest_from_kaggle(dataset_name: str, download_path: str = "Data"):
    """
    Download and extract a Kaggle dataset.
    
    Args:
        dataset_name (str): Kaggle dataset identifier (e.g., 'blastchar/telco-customer-churn')
        download_path (str): Directory where data will be saved.
    """

    os.makedirs(download_path, exist_ok=True)

    try:
        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading dataset: {dataset_name}")
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        logger.info(f"Dataset downloaded and extracted to '{download_path}'")
    except FileNotFoundError:
        logger.error(f"The download path '{download_path}' could not be created or found.")
        raise FileNotFoundError(f"The download path '{download_path}' could not be created or found.")
    except Exception as e:
        logger.exception(f"Failed to download dataset '{dataset_name}': {e}")
        raise RuntimeError(f"Failed to download dataset '{dataset_name}': {e}")