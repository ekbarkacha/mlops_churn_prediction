"""
Module: metrics.py - Centralized Monitoring Metrics for MLOps Pipeline Used in Prometheus
============================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description:
------------
This module defines all **Prometheus metrics** and the utility function to 
push monitoring data to the FastAPI service.

It integrates observability into three main components:

1. **Data Drift Checker**
   - Tracks the proportion and number of drifted features using Evidently AI.

2. **Model Decay Checker**
   - Tracks model performance metrics (accuracy, precision, recall, ROC AUC, F1)
     on recent inference data to detect model degradation.

3. **FastAPI Inference Service**
   - Tracks API request counts, latencies, errors, and model usage at inference time.

Metrics are exposed to Prometheus using the FastAPI \mentrics endpoint.

Environment Variables:
----------------------
- `PRO_TARGET_ADDRESS`: Address of the FastAPI metrics endpoint (e.g., `localhost:8000`)
- `API_KEY`: Authentication key for FastAPI metrics API

"""
import os
import requests
from prometheus_client import Counter, Histogram, Gauge
from dotenv import load_dotenv

load_dotenv()

PRO_TARGET_ADDRESS = os.getenv("PRO_TARGET_ADDRESS")
API_KEY = os.getenv("API_KEY")

#### FOR THE DATA DRIFT CHECKER PART ###
# ==============================================================================
# These metrics track how much the data distribution changes between 
# training and inference (monitored using Evidently AI reports).
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
# ==============================================================================
# Metrics used to monitor model performance degradation over time 
# (based on inference logs and actual labels).
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
# ==============================================================================
# These metrics are collected at runtime within the inference API to track
# request patterns, latency, prediction performance, and errors.

# Count of total predictions made by each model and version
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Number of predictions made by the model",
    ["model_name", "version","stage"]
)

# Histogram of prediction latency (seconds)
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Time taken to run predictions",
    ["model_name", "version","stage"]
)

# Total number of prediction errors
PREDICTION_ERRORS = Counter(
    "model_prediction_errors_total",
    "Number of prediction errors encountered",
    ["model_name", "version","stage"]
)

# Current model version loaded into production/staging
CURRENT_MODEL_VERSION = Gauge(
    "model_loaded_version",
    "Currently loaded model version number",
    ["model_name","model_type","stage"]
)

# Count total number of API requests to each endpoint
API_REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests by endpoint",
    ["endpoint", "method"]
)

# Count total failed API requests
API_ERRORS = Counter(
    "api_request_errors_total",
    "Total number of failed API requests by endpoint",
    ["endpoint", "method", "status"]
)

# Histogram of request latency per endpoint
API_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency per endpoint",
    ["method", "endpoint"]
)

### METRIC PUSHER FUNCTION ###
# ==============================================================================
# This function allows asynchronous reporting of metrics to the FastAPI service
# (e.g., for centralized metric storage or custom dashboard integration).
def push_metrics_to_fastapi(payload:dict):
    """
    Push computed metrics to the FastAPI metrics endpoint.

    Args:
        payload (dict): Metric data containing model name, drift, or decay info.
                        Example:
                        {
                            "from": "drift",
                            "model_name": "customer_churn",
                            "drift_share": 0.23,
                            "drift_count": 5
                        }
    
    Behavior:
        - Sends a POST request with the metrics payload.
        - Logs success or failure for monitoring systems.
    """
    url = f"http://{PRO_TARGET_ADDRESS}/update_metrics" 
    headers = {"x-api-key": API_KEY} 
    try:
        requests.post(url=url,headers=headers,json=payload, timeout=10)
        print("Metrics pushed to FastAPI server")
    except Exception as e:
        print("Failed to push metrics:", e)