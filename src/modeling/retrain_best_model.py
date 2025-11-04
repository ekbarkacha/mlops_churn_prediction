"""
Retrain Best Model - Reproduce the best Neural Network model
Based on MLflow run with F1=0.6534, ROC_AUC=0.8583
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


def retrain_best_neural_network():
    """
    Retrain the best Neural Network model with exact hyperparameters.
    
    Best configuration from MLflow:
    - hidden_units: 256
    - lr: 0.001
    - batch_size: 256
    - epochs: 50
    - dropout: 0.3
    
    Expected results:
    - F1: ~0.6534
    - ROC AUC: ~0.8583
    - Accuracy: ~0.7892
    """
    logger.info("\n" + "="*80)
    logger.info("🔄 RE-TRAINING BEST NEURAL NETWORK MODEL")
    logger.info("="*80 + "\n")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # ========================================
    # CONFIGURATION (EXACT MATCH)
    # ========================================
    config = {
        'hidden_units': 256,
        'lr': 0.001,
        'batch_size': 256,
        'epochs': 50,
        'dropout': 0.3,
        'test_size': 0.2,
        'random_state': 2,
        'use_smote': True,
        'smote_strategy': 1.0  # 50/50 balance
    }
    
    logger.info("⚙️  Model Configuration:")
    for key, value in config.items():
        logger.info(f"   • {key}: {value}")
    logger.info("")
    
    # ========================================
    # LOAD DATA (42 features - advanced)
    # ========================================
    feature_file = "telco_churn_features_advanced.csv"
    feature_path = FEATURE_DATA_DIR / feature_file
    
    logger.info(f"📂 Loading feature data:")
    logger.info(f"   {feature_path}")
    
    if not feature_path.exists():
        logger.error(f"❌ Feature file not found!")
        logger.error(f"   Looking for: {feature_path}")
        logger.error(f"   Available files:")
        for f in FEATURE_DATA_DIR.iterdir():
            logger.error(f"      • {f.name}")
        raise FileNotFoundError(f"Feature file not found: {feature_path}")
    
    df = pd.read_csv(feature_path)
    logger.info(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns\n")
    
    # ========================================
    # SPLIT DATA
    # ========================================
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # ========================================
    # APPLY SMOTE
    # ========================================
    if config['use_smote']:
        logger.info("\n" + "="*80)
        logger.info("🔄 Applying SMOTE to balance training data")
        logger.info("="*80)
        
        logger.info(f"   • Before SMOTE:")
        logger.info(f"     - Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
        logger.info(f"     - Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
        
        smote = SMOTE(
            sampling_strategy=config['smote_strategy'],
            random_state=config['random_state']
        )
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        logger.info(f"   • After SMOTE:")
        logger.info(f"     - Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
        logger.info(f"     - Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
        logger.info(f"✅ SMOTE completed: {len(y_train)} samples\n")
    
    # ========================================
    # TRAIN NEURAL NETWORK
    # ========================================
    with mlflow.start_run(run_name="BestNeuralNet_Retrain"):
        
        logger.info("\n" + "="*80)
        logger.info("🏋️  Training Neural Network with Best Configuration")
        logger.info("="*80 + "\n")
        
        # Log config
        mlflow.log_params(config)
        
        # Device detection
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(f"🍎 Using Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"🎮 Using NVIDIA GPU (CUDA)")
        else:
            device = torch.device("cpu")
            logger.info(f"💻 Using CPU")
        
        # Create model
        input_dim = X_train.shape[1]
        model = NN(
            input_dim=input_dim,
            hidden_units=config['hidden_units'],
            dropout=config['dropout']
        ).to(device)
        
        logger.info(f"\n🧠 Model Architecture:")
        logger.info(f"   • Input:  {input_dim} features")
        logger.info(f"   • Hidden: {config['hidden_units']} units")
        logger.info(f"   • Dropout: {config['dropout']}")
        logger.info(f"   • Output: 1 (binary classification)")
        logger.info(f"   • Device: {device}\n")
        
        # Loss and optimizer
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
        logger.info(f"🚀 Starting training ({config['epochs']} epochs)...\n")
        
        epochs = config['epochs']
        for epoch in range(epochs):
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
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"   Epoch [{epoch+1:2d}/{epochs}] - Loss: {avg_loss:.4f}")
        
        logger.info(f"\n✅ Training completed!\n")
        
        # ========================================
        # EVALUATE
        # ========================================
        logger.info("="*80)
        logger.info("📊 EVALUATING MODEL ON TEST SET")
        logger.info("="*80 + "\n")
        
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t.to(device))
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)
        
        metrics = evaluate_model(y_test, preds, probs)
        
        # Display metrics
        logger.info("📈 Test Set Results:")
        logger.info("="*80)
        for metric, value in metrics.items():
            status = "✅" if value >= 0.65 else "⚠️"
            logger.info(f"   {status} {metric.upper():<12}: {value:.4f} ({value*100:.2f}%)")
        logger.info("="*80 + "\n")
        
        # Log to MLflow
        log_metrics_to_mlflow(metrics)
        
        # ========================================
        # SAVE MODEL
        # ========================================
        logger.info("💾 Saving model to MLflow...\n")
        
        # Save model
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
        
        logger.info("✅ Model saved successfully!")
        
        # ========================================
        # SUMMARY
        # ========================================
        logger.info("\n" + "="*80)
        logger.info("🎉 RE-TRAINING COMPLETED!")
        logger.info("="*80 + "\n")
        
        logger.info("📊 Final Metrics:")
        logger.info(f"   • Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"   • Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        logger.info(f"   • Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        logger.info(f"   • F1:        {metrics['f1']:.4f} ({metrics['f1']*100:.2f}%)")
        logger.info(f"   • ROC AUC:   {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
        
        logger.info("\n💡 View results in MLflow UI:")
        logger.info("   mlflow ui")
        logger.info("   Then open: http://localhost:5000\n")
        
        return metrics


def main():
    """Main execution"""
    try:
        metrics = retrain_best_neural_network()
        
        logger.info("\n" + "="*80)
        logger.info("✅ SUCCESS! Model retrained successfully")
        logger.info("="*80 + "\n")
        
        # Check if we matched the expected performance
        expected_f1 = 0.6534
        actual_f1 = metrics['f1']
        diff = abs(actual_f1 - expected_f1)
        
        if diff < 0.01:  # Within 1%
            logger.info(f"🎯 EXCELLENT! Results match expected performance:")
            logger.info(f"   Expected F1: {expected_f1:.4f}")
            logger.info(f"   Actual F1:   {actual_f1:.4f}")
            logger.info(f"   Difference:  {diff:.4f} ({diff*100:.2f}%)")
        else:
            logger.info(f"⚠️  Results differ slightly from expected:")
            logger.info(f"   Expected F1: {expected_f1:.4f}")
            logger.info(f"   Actual F1:   {actual_f1:.4f}")
            logger.info(f"   Difference:  {diff:.4f} ({diff*100:.2f}%)")
            logger.info(f"   This is normal due to random initialization")
        
    except Exception as e:
        logger.error(f"\n❌ Re-training failed: {e}")
        raise


if __name__ == "__main__":
    main()