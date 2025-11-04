"""
Production Model - Best Neural Network (F1=64.51%)
This is the FINAL production-ready model
"""

import sys
from pathlib import Path
import yaml
import mlflow
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.const import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_CONFIG_PATH,
    FEATURE_DATA_DIR,
    PREPROCESSORS_DIR
)
from src.modeling.model_utils import split_data, evaluate_model, log_metrics_to_mlflow
from src.modeling.nn_model import NN, PyTorchWrapper

logger = get_logger(__name__)


def train_production_model():
    """
    Train the PRODUCTION model with optimal configuration.
    
    Configuration:
    - Architecture: NN(42 → 256 → 1)
    - Learning Rate: 0.001
    - Batch Size: 256
    - Epochs: 50
    - Dropout: 0.3
    - SMOTE: 50/50 balance
    
    Expected Performance:
    - F1: 64.51%
    - Precision: 58.48%
    - Recall: 71.93%
    - ROC AUC: 85.35%
    - Accuracy: 78.99%
    """
    
    logger.info("\n" + "="*80)
    logger.info("🚀 TRAINING PRODUCTION MODEL")
    logger.info("="*80 + "\n")
    
    # Load config
    with open(MODEL_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    nn_config = config['neural_net']
    train_config = config['training']
    prod_config = config['production']
    
    logger.info("⚙️  Production Configuration:")
    logger.info(f"   • Model: {prod_config['model_name']}")
    logger.info(f"   • Version: {prod_config['model_version']}")
    logger.info(f"   • Architecture: NN(42 → {nn_config['hidden_units']} → 1)")
    logger.info(f"   • Learning Rate: {nn_config['lr']}")
    logger.info(f"   • Batch Size: {nn_config['batch_size']}")
    logger.info(f"   • Epochs: {nn_config['epochs']}")
    logger.info(f"   • Dropout: {nn_config['dropout']}\n")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Load data
    feature_path = FEATURE_DATA_DIR / prod_config['feature_file']
    logger.info(f"📂 Loading features: {feature_path}")
    df = pd.read_csv(feature_path)
    logger.info(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # Split
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=train_config['test_size'],
        random_state=train_config['random_state']
    )
    
    # SMOTE
    if train_config['use_smote']:
        logger.info("🔄 Applying SMOTE (50/50 balance)...")
        smote = SMOTE(
            sampling_strategy=train_config['smote_strategy'],
            random_state=train_config['smote_random_state']
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        logger.info(f"✅ SMOTE completed: {len(y_train)} samples\n")
    
    # Train
    with mlflow.start_run(run_name=f"{prod_config['model_name']}_v{prod_config['model_version']}"):
        
        # Log config
        mlflow.log_params({
            'model_version': prod_config['model_version'],
            'hidden_units': nn_config['hidden_units'],
            'lr': nn_config['lr'],
            'batch_size': nn_config['batch_size'],
            'epochs': nn_config['epochs'],
            'dropout': nn_config['dropout']
        })
        
        # Device
        device = torch.device("cpu")
        logger.info(f"💻 Device: {device}\n")
        
        # Model
        input_dim = X_train.shape[1]
        model = NN(input_dim, nn_config['hidden_units'], nn_config['dropout']).to(device)
        
        logger.info("🏋️  Training...\n")
        
        # Loss & Optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=nn_config['lr'])
        
        # Data
        X_train_t = torch.from_numpy(X_train.to_numpy(dtype=np.float32))
        y_train_t = torch.from_numpy(y_train.to_numpy(dtype=np.float32).reshape(-1, 1))
        X_test_t = torch.from_numpy(X_test.to_numpy(dtype=np.float32))
        
        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=nn_config['batch_size'],
            shuffle=True
        )
        
        # Training loop
        for epoch in range(nn_config['epochs']):
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
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"   Epoch [{epoch+1}/{nn_config['epochs']}] - Loss: {epoch_loss/len(train_loader):.4f}")
        
        logger.info("\n✅ Training completed!\n")
        
        # Evaluate
        logger.info("="*80)
        logger.info("📊 PRODUCTION MODEL EVALUATION")
        logger.info("="*80 + "\n")
        
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > prod_config['threshold']).astype(int)
        
        metrics = evaluate_model(y_test, preds, probs)
        
        # Display
        logger.info("📈 Production Model Performance:")
        for metric, value in metrics.items():
            status = "✅" if value >= 0.60 else "⚠️"
            logger.info(f"   {status} {metric.upper():<12}: {value:.4f} ({value*100:.2f}%)")
        logger.info("")
        
        log_metrics_to_mlflow(metrics)
        
        # Save
        logger.info("💾 Saving production model...")
        model_cpu = model.to("cpu")
        input_example = X_test.iloc[:2].to_numpy().astype(np.float64).tolist()
        
        mlflow.pyfunc.log_model(
            python_model=PyTorchWrapper(model_cpu),
            artifact_path="model",
            input_example=input_example
        )
        
        # Log preprocessors
        for file_name in PREPROCESSORS_DIR.iterdir():
            if file_name.is_file():
                mlflow.log_artifact(str(file_name), artifact_path="preprocessors")
        
        logger.info("✅ Model saved to MLflow!")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("🎉 PRODUCTION MODEL READY!")
        logger.info("="*80 + "\n")
        
        logger.info("📊 Final Metrics:")
        logger.info(f"   • F1 Score:  {metrics['f1']:.4f} (Target: 0.6451)")
        logger.info(f"   • ROC AUC:   {metrics['roc_auc']:.4f} (Target: 0.8535)")
        logger.info(f"   • Precision: {metrics['precision']:.4f}")
        logger.info(f"   • Recall:    {metrics['recall']:.4f}")
        logger.info(f"   • Accuracy:  {metrics['accuracy']:.4f}")
        
        logger.info("\n💡 Model is production-ready for deployment!")
        logger.info("   Next: Deploy via API/Batch/Dashboard\n")
        
        return metrics


def main():
    """Main execution"""
    try:
        metrics = train_production_model()
        
        logger.info("="*80)
        logger.info("✅ PRODUCTION MODEL TRAINING COMPLETED")
        logger.info("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Production model training failed: {e}")
        raise


if __name__ == "__main__":
    main()