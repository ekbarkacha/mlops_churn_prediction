"""
Configuration module - Loads environment variables and Kaggle credentials
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Kaggle Credentials
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Validate credentials
if not KAGGLE_USERNAME or not KAGGLE_KEY:
    raise ValueError(
        "Kaggle credentials not found! Please set KAGGLE_USERNAME and KAGGLE_KEY in .env file"
    )

# Dataset
KAGGLE_DATASET = os.getenv("KAGGLE_DATASET", "blastchar/telco-customer-churn")

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
LOGS_DIR = PROJECT_ROOT / os.getenv("LOGS_DIR", "logs")

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Display config (for debugging)
if __name__ == "__main__":
    print(f"✅ Kaggle Username: {KAGGLE_USERNAME}")
    print(f"✅ Kaggle Dataset: {KAGGLE_DATASET}")
    print(f"✅ Raw Data Dir: {RAW_DATA_DIR}")
    print(f"✅ Processed Data Dir: {PROCESSED_DATA_DIR}")
    print(f"✅ Logs Dir: {LOGS_DIR}")