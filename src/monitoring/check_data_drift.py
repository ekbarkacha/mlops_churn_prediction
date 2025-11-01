import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import json
from evidently import Report
from evidently.presets import DataDriftPreset
from src.utils.logger import get_logger
from src.utils.const import (EXPECTED_COLUMNS,RAW_DATA_DIR,raw_file_name,
                             INFERENCE_DATA_DIR,inference_file_name,
                             REPORTS_DIR,data_drift_report_json,data_drift_report_html,
                             MLFLOW_EXPERIMENT_NAME)
from src.utils.config import DRIFT_THRESHOLD
from src.data_pipeline.data_preprocessing import data_cleaning,validate_schema
from src.monitoring.metrics import push_metrics_to_fastapi
from src.data_pipeline.data_ingestion import ingest_from_csv

logger = get_logger(__name__)

BASELINE_PATH = os.path.join(RAW_DATA_DIR, raw_file_name)
NEW_DATA_PATH = os.path.join(INFERENCE_DATA_DIR, inference_file_name)
DRIFT_REPORT_JSON_PATH = os.path.join(REPORTS_DIR, data_drift_report_json)
DRIFT_REPORT_HTML_PATH = os.path.join(REPORTS_DIR, data_drift_report_html)

def extract_drift_info(result_dict):
    """
    Extract drift metrics
    """
    try:
        for metric in result_dict.get("metrics", []):
            if metric.get("metric_id", "").startswith("DriftedColumnsCount"):
                value = metric.get("value", {})
                drift_share = float(value.get("share", 0))
                drift_count = int(value.get("count", 0))
                return drift_share, drift_count
    except Exception as e:
        logger.error(f"Failed to parse drift info: {e}")
    return None, None


def check_drift():
    logger.info("Checking for data drift using Evidently ...")

    # Load data
    baseline = ingest_from_csv(BASELINE_PATH)
    new_data = ingest_from_csv(NEW_DATA_PATH)

    baseline = baseline[EXPECTED_COLUMNS]
    new_data = new_data[EXPECTED_COLUMNS]

    # Preprocessing
    baseline = data_cleaning(baseline)
    #baseline = encode_categorical(baseline)
    #baseline = create_features(baseline)

    # Preprocessing
    new_data = data_cleaning(new_data)
    #new_data = encode_categorical(new_data)
    #new_data = create_features(new_data)

    # Generate Evidently drift report
    report = Report(metrics=[DataDriftPreset()], include_tests=True)
    report = report.run(reference_data=baseline, current_data=new_data)

    # Convert to dict and save 
    result = report.dict()
    os.makedirs(os.path.dirname(DRIFT_REPORT_JSON_PATH), exist_ok=True)
    with open(DRIFT_REPORT_JSON_PATH, "w") as f:
        json.dump(result, f, indent=4)

    # Extract drift metrics 
    drift_share, drift_count = extract_drift_info(result)

    if drift_count is None:
        logger.error("Could not extract drift metrics from Evidently report.")
        logger.debug(f"Full report preview: {json.dumps(result, indent=2)[:1000]}")
        push_metrics_to_fastapi({"from":"pipeline_error","pipeline_name":"drift_check"})
        print("Could not extract drift metrics from Evidently report.")
        return 1  # assume drift occurred
   
    push_metrics_to_fastapi({"from":"drift","model_name":MLFLOW_EXPERIMENT_NAME,"drift_share": drift_share, "drift_count": drift_count})


    logger.info(f"Drifted columns: {drift_count}, Drift share: {drift_share:.3f}")
    drift_detected = drift_share > DRIFT_THRESHOLD
    logger.info(f"Drift detected: {drift_detected}")
    print(f"Drift detected: {drift_detected}")

    # Save HTML report for visualization
    report.save_html(DRIFT_REPORT_HTML_PATH)
    logger.info(f"Drift report saved to: {DRIFT_REPORT_HTML_PATH}")

    return 1 if drift_detected else 0


if __name__ == "__main__":
    exit_code = check_drift()
    print(f"Drift check completed with exit code {exit_code}",flush=True)
    sys.exit(exit_code)

