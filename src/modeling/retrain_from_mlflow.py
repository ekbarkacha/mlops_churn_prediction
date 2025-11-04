"""
Retrain Best Model from MLflow Run ID
Load exact configuration from MLflow and retrain the model
"""
import sys
from pathlib import Path
import mlflow
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.const import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    FEATURE_DATA_DIR,
    PREPROCESSORS_DIR
)
from src.modeling.model_utils import split_data, evaluate_model, log_metrics_to_mlflow
from src.modeling.nn_model import NN, PyTorchWrapper

logger = get_logger(__name__)


def load_run_config(run_id: str) -> dict:
    """
    Load configuration from MLflow run.
    
    Args:
        run_id (str): MLflow run ID
    
    Returns:
        dict: Run configuration
    """
    logger.info(f"📂 Loading configuration from MLflow Run: {run_id}")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    
    # Get run
    run = client.get_run(run_id)
    
    # Extract params
    params = run.data.params
    metrics = run.data.metrics
    
    logger.info(f"\n✅ Run found: {run.info.run_name}")
    logger.info(f"   • Start time: {run.info.start_time}")
    logger.info(f"   • Status: {run.info.status}")
    
    logger.info(f"\n📊 Original Metrics:")
    for metric, value in metrics.items():
        logger.info(f"   • {metric}: {value:.4f}")
    
    logger.info(f"\n⚙️  Original Parameters:")
    for param, value in params.items():
        logger.info(f"   • {param}: {value}")
    
    return {
        'run_name': run.info.run_name,
        'params': params,
        'metrics': metrics
    }


def retrain_from_mlflow_run(run_id: str = "34e2a57ef6104819aa049946fb8cca2d"):
    """
    Retrain the exact model from MLflow run.
    
    Args:
        run_id (str): MLflow run ID to replicate
    """
    
    logger.info("\n" + "="*80)
    logger.info("🔄 RE-TRAINING BEST MODEL FROM MLFLOW RUN")
    logger.info("="*80 + "\n")
    
    # Load run config
    run_config = load_run_config(run_id)
    params = run_config['params']
    original_metrics = run_config['metrics']
    
    # Parse parameters
    config = {
        'hidden_units': int(params.get('hidden_units', 256)),
        'lr': float(params.get('lr', 0.001)),
        'batch_size': int(params.get('batch_size', 256)),
        'epochs': int(params.get('epochs', 50)),
        'dropout': float(params.get('dropout', 0.3)),
        'test_size': 0.2,
        'random_state': 2,
        'use_smote': True,
        'smote_strategy': 1.0
    }
    
    logger.info("\n" + "="*80)
    logger.info("⚙️  REPLICATION CONFIGURATION")
    logger.info("="*80)
    for key, value in config.items():
        logger.info(f"   • {key}: {value}")
    logger.info("")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Load data
    feature_path = FEATURE_DATA_DIR / "telco_churn_features_advanced.csv"
    
    logger.info(f"📂 Loading features: {feature_path}")
    
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    
    df = pd.read_csv(feature_path)
    logger.info(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # Split data (same random state as original)
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # Apply SMOTE
    if config['use_smote']:
        logger.info("="*80)
        logger.info("🔄 Applying SMOTE (50/50 balance)")
        logger.info("="*80)
        logger.info(f"   • Before: Class 0={sum(y_train==0)}, Class 1={sum(y_train==1)}")
        
        smote = SMOTE(
            sampling_strategy=config['smote_strategy'],
            random_state=config['random_state']
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        logger.info(f"   • After: Class 0={sum(y_train==0)}, Class 1={sum(y_train==1)}")
        logger.info(f"✅ SMOTE completed: {len(y_train)} samples\n")
    
    # Train with MLflow tracking
    with mlflow.start_run(run_name=f"Retrain_Best_{run_id[:8]}"):
        
        logger.info("="*80)
        logger.info("🏋️  TRAINING MODEL")
        logger.info("="*80 + "\n")
        
        # Log config
        mlflow.log_params(config)
        mlflow.log_param('source_run_id', run_id)
        
        # Device
        device = torch.device("cpu")
        logger.info(f"💻 Device: {device}\n")
        
        # Model
        input_dim = X_train.shape[1]
        model = NN(
            input_dim=input_dim,
            hidden_units=config['hidden_units'],
            dropout=config['dropout']
        ).to(device)
        
        logger.info(f"🧠 Model Architecture:")
        logger.info(f"   • Input:  {input_dim} features")
        logger.info(f"   • Hidden: {config['hidden_units']} units")
        logger.info(f"   • Dropout: {config['dropout']}")
        logger.info(f"   • Output: 1 (binary)\n")
        
        # Loss & Optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        # Prepare data
        X_train_t = torch.from_numpy(X_train.to_numpy(dtype=np.float32))
        y_train_t = torch.from_numpy(y_train.to_numpy(dtype=np.float32).reshape(-1, 1))
        X_test_t = torch.from_numpy(X_test.to_numpy(dtype=np.float32))
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=config['batch_size'],
            shuffle=True
        )
        
        # Training loop
        logger.info(f"🚀 Training {config['epochs']} epochs...\n")
        
        for epoch in range(config['epochs']):
            model.train()
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"   Epoch [{epoch+1:2d}/{config['epochs']}] - Loss: {avg_loss:.4f}")
        
        logger.info(f"\n✅ Training completed!\n")
        
        # Evaluate
        logger.info("="*80)
        logger.info("📊 MODEL EVALUATION")
        logger.info("="*80 + "\n")
        
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
        
        metrics = evaluate_model(y_test, preds, probs)
        
        # Display results
        logger.info("📈 COMPARISON - Original vs Retrained:")
        logger.info("="*80)
        logger.info(f"{'Metric':<15} {'Original':<15} {'Retrained':<15} {'Diff':<15}")
        logger.info("-"*80)
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            original = original_metrics.get(metric_name, 0.0)
            retrained = metrics.get(metric_name, 0.0)
            diff = retrained - original
            diff_str = f"{diff:+.4f}" if diff != 0 else "0.0000"
            
            status = "✅" if abs(diff) < 0.02 else "⚠️"
            logger.info(f"{status} {metric_name.upper():<14} {original:.4f}         {retrained:.4f}         {diff_str}")
        
        logger.info("="*80 + "\n")
        
        # Log metrics
        log_metrics_to_mlflow(metrics)
        
        # Check if replication successful
        f1_diff = abs(metrics['f1'] - original_metrics.get('f1', 0.0))
        
        if f1_diff < 0.02:  # Within 2%
            logger.info("🎯 SUCCESS! Model successfully replicated!")
            logger.info(f"   F1 difference: {f1_diff:.4f} (< 2%)\n")
        else:
            logger.info("⚠️  Model differs slightly from original")
            logger.info(f"   F1 difference: {f1_diff:.4f}")
            logger.info("   This is normal due to random initialization\n")
        
        # Save model
        logger.info("💾 Saving model to MLflow...")
        
        model_cpu = model.to("cpu")
        input_example = X_test.iloc[:2].to_numpy().astype(np.float64).tolist()
        
        mlflow.pyfunc.log_model(
            python_model=PyTorchWrapper(model_cpu),
            artifact_path="model",
            input_example=input_example
        )
        
        # Log preprocessors
        logger.info("📦 Logging preprocessors...")
        for file_name in PREPROCESSORS_DIR.iterdir():
            if file_name.is_file():
                mlflow.log_artifact(str(file_name), artifact_path="preprocessors")
        
        logger.info("✅ Model saved successfully!\n")
        
        # Final summary
        logger.info("="*80)
        logger.info("🎉 RE-TRAINING COMPLETED!")
        logger.info("="*80 + "\n")
        
        logger.info("📊 Final Metrics:")
        logger.info(f"   • Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"   • Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        logger.info(f"   • Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        logger.info(f"   • F1 Score:  {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        logger.info(f"   • ROC AUC:   {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
        
        logger.info("\n🎯 Target Metrics (from original run):")
        logger.info(f"   • F1 Score:  {original_metrics.get('f1', 0):.4f}")
        logger.info(f"   • ROC AUC:   {original_metrics.get('roc_auc', 0):.4f}")
        
        logger.info("\n💡 View results in MLflow UI:")
        logger.info("   mlflow ui")
        logger.info("   Then open: http://localhost:5000\n")
        
        return metrics


def main():
    """Main execution"""
    
    # Best run ID from your MLflow
    BEST_RUN_ID = "34e2a57ef6104819aa049946fb8cca2d"
    
    logger.info("="*80)
    logger.info("🎯 RETRAIN BEST MODEL FROM MLFLOW")
    logger.info("="*80 + "\n")
    logger.info(f"Run ID: {BEST_RUN_ID}\n")
    
    try:
        metrics = retrain_from_mlflow_run(BEST_RUN_ID)
        
        logger.info("="*80)
        logger.info("✅ RETRAINING SUCCESSFUL!")
        logger.info("="*80 + "\n")
        
        logger.info("📦 Model saved and ready for production!")
        logger.info("   Next steps:")
        logger.info("   1. Register model in MLflow")
        logger.info("   2. Deploy via API/Batch/Dashboard")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()