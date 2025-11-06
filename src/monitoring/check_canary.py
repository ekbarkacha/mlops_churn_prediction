"""
Module: check_canary.py
=======================

Author: AIMS-AMMI STUDENT 
Created: October/November 2025  
Description: 
------------
This module evaluates the performance of **canary (Staging)** and **production** models 
based on logged inference data. It helps decide whether the canary model should be 
**promoted to production** or **kept in staging**.

The evaluation compares key metrics (e.g., F1-score, latency) and applies composite 
scoring to ensure that the canary model is both more accurate and efficient.

Workflow Summary:
-----------------
1. Load inference logs from CSV.
2. Validate presence of essential columns.
3. Compute evaluation metrics (via `evaluate_model`) for each model stage.
4. Compare F1 scores and latencies using a weighted composite score.
5. Return a structured decision dict indicating whether promotion should occur.

"""    

# Imports and setup
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pandas as pd

# Custom utility imports
from src.modeling.model_utils import evaluate_model
from src.utils.logger import get_logger
from src.utils.const import INFERENCE_DATA_DIR, inference_file_name

# Initialize logger
logger = get_logger(__name__)

def check_canary_from_logs(min_samples=30):
    """
    Evaluate canary (Staging) vs Production models based on logged inference data.
    Only promote canary if it outperforms production and meets minimum sample size.
    
    Args:
        min_samples (int): minimum number of predictions per stage required for evaluation
    Returns:
        dict: evaluation result including metrics and promotion decision
    """
    logger.info("Evaluating Production vs Staging from inference logs...")

    # Load logged inference data
    path = f"{INFERENCE_DATA_DIR}/{inference_file_name}"
    df = pd.read_csv(path)

    # Ensure required columns exist
    required_cols = ["Churn", "probability", "model_stage", "latency"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Map target label if needed
    if df["Churn"].dtype == "object":
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
        logger.info("Converted target labels from string to binary integers.")


    # Compute metrics for each model stage
    results = {}
    for stage, group in df.groupby("model_stage"):
        if len(group) < min_samples:
            logger.warning(f"Stage '{stage}' has only {len(group)} samples, below min_samples={min_samples}. Skipping.")
            continue

        y_true = group["Churn"].values
        y_prob = group["probability"].values
        y_pred = (y_prob > 0.5).astype(int)
        metrics = evaluate_model(y_true, y_pred, y_prob)
        metrics["mean_latency"] = group["latency"].mean()
        metrics["num_samples"] = len(group)
        results[stage] = metrics
        logger.info(f"{stage} metrics: {metrics}")

    # Thresholds and scoring
    MODEL_THRESHOLD = 0.01  # Require 1% improvement in F1 to promote
    LATENCY_IMPORTANCE = 0.1
    MAX_ALLOWED_LATENCY = 2.0  # seconds

    prod_metrics = results.get("Production", {})
    stag_metrics = results.get("Staging", {})

    prod_f1 = prod_metrics.get("f1", 0)
    stag_f1 = stag_metrics.get("f1", 0)
    prod_latency = prod_metrics.get("mean_latency", 0)
    stag_latency = stag_metrics.get("mean_latency", 0)

    # Composite score (F1 adjusted for latency)
    prod_score = prod_f1 - LATENCY_IMPORTANCE * prod_latency
    stag_score = stag_f1 - LATENCY_IMPORTANCE * stag_latency

    logger.info(f"Production Score={prod_score:.4f}, Staging Score={stag_score:.4f}")

    # Promotion decision
    promote = (
        (stag_score > prod_score + MODEL_THRESHOLD)
        and (stag_latency <= MAX_ALLOWED_LATENCY)
        and (stag_metrics.get("num_samples", 0) >= min_samples)
    )
    better_model = "Staging" if promote else "Production"

    logger.info(f"Better model: {better_model}, Promote: {promote}")

    return {
        "promote": promote,
        "better_model": better_model,
        "canary_metrics": stag_metrics,
        "production_metrics": prod_metrics,
        "prod_score": prod_score,
        "stag_score": stag_score
    }


if __name__ == "__main__":
    """
    Entry point for manual execution during testing.
    """
    print(check_canary_from_logs())