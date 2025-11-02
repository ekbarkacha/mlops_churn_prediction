import os
from dotenv import load_dotenv

load_dotenv()

#Setup kaggle
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Setup Mlflow uri (note its aslo in src/app/config)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Drift Threshold
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD",0.5))

# Model decay baseline
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD",0.80))