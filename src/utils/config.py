import os
from dotenv import load_dotenv

load_dotenv()

#Setup kaggle
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")

# Setup Mlflow uri (note its aslo in src/app/config)
if os.getenv("RUNNING_IN_DOCKER"):
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
else:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_LOCAL")

# Drift Threshold
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD",0.5))

# Model decay baseline
MODEL_THRESHOLD = float(os.getenv("MODEL_THRESHOLD",0.80))