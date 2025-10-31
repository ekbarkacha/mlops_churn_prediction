import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.const import (USERS_DATA_DIR,
                             users_file_name,
                             INFERENCE_DATA_DIR,
                             inference_file_name,
                             MLFLOW_EXPERIMENT_NAME,
                             MODEL_DIR,EXPECTED_COLUMNS,
                             PROCESSED_DATA_DIR,processed_file_name,
                             RAW_DATA_DIR,raw_file_name)
from dotenv import load_dotenv

load_dotenv()

#FastApi
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
API_KEY = os.getenv("API_KEY")
ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("ACCESS_TOKEN_EXPIRE_DAYS", 30))

# Setup Mlflow uri (note its aslo in src/utils/config)
if os.getenv("RUNNING_IN_DOCKER"):
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
else:
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_LOCAL")

USERS_FILE = f"{USERS_DATA_DIR}/{users_file_name}"

APP_USERS = {1:"agent",2:"admin"} 

INFERENCE_DATA_PATH = f"{INFERENCE_DATA_DIR}/{inference_file_name}"

PROCESSED_PATH = f"{PROCESSED_DATA_DIR}/{processed_file_name}"

RAW_DATA_PATH = f"{RAW_DATA_DIR}/{raw_file_name}"



