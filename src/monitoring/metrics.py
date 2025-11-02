import os
import requests
from prometheus_client import Counter, Histogram, Gauge
from dotenv import load_dotenv

load_dotenv()

PRO_TARGET_ADDRESS = os.getenv("PRO_TARGET_ADDRESS")
API_KEY = os.getenv("API_KEY")

#### FOR THE DATA DRIFT CHECKER PART ###
DRIFT_SHARE_GAUGE = Gauge(
    'ml_drift_share',
    'Share of features drifted',
    ['model_name']
)

DRIFT_COUNT_GAUGE = Gauge(
    'ml_drift_count',
    'Number of features drifted',
    ['model_name']
)

#### FOR THE MODEL DECAY CHECKER PART ###
MODEL_ACCURACY_GAUGE = Gauge(
    'ml_model_accuracy',
    'Accuracy score on recent inference',
    ['model_name', "model_type", 'version']
)

MODEL_PRECISION_GAUGE = Gauge(
    'ml_model_precision',
    'Precision score on recent inference',
    ['model_name', "model_type", 'version']
)

MODEL_RECALL_GAUGE = Gauge(
    'ml_model_recall',
    'Recall score on recent inference',
    ['model_name', "model_type", 'version']
)
MODEL_ROC_AUC_GAUGE = Gauge(
    'ml_model_roc_auc',
    'roc_auc score on recent inference',
    ['model_name', "model_type", 'version']
)
MODEL_F1_GAUGE = Gauge(
    'ml_model_f1',
    'F1 score on recent inference',
    ['model_name', "model_type", 'version']
)

PIPELINE_ERROR_COUNTER = Counter(
    'ml_pipeline_errors_total',
    'Total pipeline errors',
    ['pipeline_name']
)


#### FOR THE FASTAPI PART ###

# Track total predictions made
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Number of predictions made by the model",
    ["model_name", "version"]
)

# Track prediction latency
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time taken to run predictions",
    ["model_name", "version"]
)

# Track number of prediction errors
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Number of prediction errors encountered",
    ["model_name", "version"]
)

# Track loaded model version
CURRENT_MODEL_VERSION = Gauge(
    "model_loaded_version",
    "Currently loaded model version number",
    ["model_name","model_type"]
)

# API request counters for each endpoint
API_REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests by endpoint",
    ["endpoint", "method"]
)

# API error counters
API_ERRORS = Counter(
    "api_request_errors_total",
    "Total number of failed API requests by endpoint",
    ["endpoint", "method", "status"]
)

# API latency histogram
API_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency per endpoint",
    ["method", "endpoint"]
)

def push_metrics_to_fastapi(payload:dict):
    url = f"http://{PRO_TARGET_ADDRESS}/update_metrics" 
    headers = {"x-api-key": API_KEY} 
    try:
        requests.post(url=url,headers=headers,json=payload)
        print("Metrics pushed to FastAPI server")
    except Exception as e:
        print("Failed to push metrics:", e)