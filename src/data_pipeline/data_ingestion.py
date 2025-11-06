"""
Module: data_ingestion.py
==========================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description: 
------------
This module provides utilities for ingesting datasets from different sources 
such as local CSV files and Kaggle datasets. It handles file existence checks, 
error logging and exceptions to ensure robust data loading.

Functions:
- ingest_from_csv(path: str) -> pd.DataFrame:
    Loads data from a local CSV file into a pandas DataFrame.

- ingest_from_kaggle(dataset_name: str, download_path: str = "Data"):
    Downloads and extracts a Kaggle dataset to a specified directory.

"""
# Imports and setup
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd

# Custom utility imports
from src.utils.logger import get_logger
from src.utils.config import KAGGLE_USERNAME,KAGGLE_KEY

# Set Kaggle API credentials as environment variables
os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY
    
from kaggle.api.kaggle_api_extended import KaggleApi

logger = get_logger(__name__)

def ingest_from_csv(path: str) -> pd.DataFrame:
    """
    Load data from a local CSV file into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the CSV is empty or cannot be parsed.
        RuntimeError: For unexpected errors while reading the file.
    """
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
    Download and extract a dataset from Kaggle.

    Args:
        dataset_name (str): Kaggle dataset identifier 
                            (e.g., 'blastchar/telco-customer-churn').
        download_path (str, optional): Directory where the dataset will be saved. Defaults to "Data".

    Raises:
        FileNotFoundError: If the download path cannot be created.
        RuntimeError: If downloading the dataset fails.
    """
    # Ensure the download directory exists
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