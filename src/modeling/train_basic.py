"""
Train model with BASIC features (13 features only)
For API compatibility with correct data types
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import mlflow
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path("data/features")
MLFLOW_URI = "file:///C:/Users/pc/Desktop/mlops_churn_prediction/mlruns"
EXPERIMENT_NAME = "churn_prediction"

# Features to use (13 features - API compatible)
BASIC_FEATURES = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]

def main():
    logger.info("=" * 80)
    logger.info("🚀 Training BASIC Model (13 features - API Compatible)")
    logger.info("=" * 80)
    
    # Load data
    df = pd.read_csv(DATA_DIR / "telco_churn_features_advanced.csv")
    logger.info(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # Select only basic features
    df_basic = df[BASIC_FEATURES + ['Churn']].copy()
    logger.info(f"✅ Selected {len(BASIC_FEATURES)} basic features")
    
    # === CONVERSION DES TYPES POUR API ===
    # Float columns (scaled values - MUST be float for API)
    float_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in float_cols:
        df_basic[col] = df_basic[col].astype('float64')
    
    # Integer columns (categorical encoded)
    int_cols = ['SeniorCitizen', 'Partner', 'Dependents', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                'Contract', 'PaperlessBilling', 'PaymentMethod']
    for col in int_cols:
        df_basic[col] = df_basic[col].astype('int64')
    
    logger.info(f"✅ Data types: {df_basic.dtypes.value_counts().to_dict()}")
    # === FIN CONVERSION ===
    
    # Split
    X = df_basic.drop('Churn', axis=1)
    y = df_basic['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    logger.info(f"✅ SMOTE applied: {len(X_train_balanced)} samples")
    
    # Train XGBoost
    logger.info("🏋️  Training XGBoost (Basic Features)...")
    
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="XGBoost_Basic_API_v2"):
        model = XGBClassifier(
            max_depth=3,
            learning_rate=0.02,
            n_estimators=300,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1]),
            min_child_weight=9
        )
        
        model.fit(X_train_balanced, y_train_balanced)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 MODEL PERFORMANCE (Basic Features - API Compatible)")
        logger.info("=" * 80)
        for metric, value in metrics.items():
            logger.info(f"   {metric.upper():12s}: {value:.4f} ({value*100:.2f}%)")
            mlflow.log_metric(metric, value)
        logger.info("=" * 80)
        
        # Log params
        mlflow.log_param("n_features", len(BASIC_FEATURES))
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("api_compatible", True)
        
        # Log model with correct schema
        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=X_test.iloc[:1]  # This will infer correct types
        )
        
        run_id = mlflow.active_run().info.run_id
        logger.info(f"\n✅ Model saved! Run ID: {run_id}")
        logger.info(f"✅ Data types preserved for API compatibility")
    
    logger.info("\n🎉 Training completed! Restart the API to use this model.")

if __name__ == "__main__":
    main()