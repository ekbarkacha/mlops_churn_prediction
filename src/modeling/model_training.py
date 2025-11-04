"""
Model Training Module - Train and evaluate multiple models with MLflow tracking
"""
import sys
from pathlib import Path
import yaml
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

  

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.const import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_CONFIG_PATH,
    FEATURE_DATA_DIR,
    FEATURE_FILE_NAME,
    PREPROCESSORS_DIR
)
from src.modeling.model_utils import (
    split_data,
    evaluate_model,
    log_metrics_to_mlflow,
    WrappedModel
)
from src.modeling.nn_model import NN, PyTorchWrapper

logger = get_logger(__name__)

# Load model configuration
with open(MODEL_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)


def tune_and_log_model(
    model_cls,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    params: dict,
    all_metrics: dict,
    log_model_func
):
    """
    Train model with optional hyperparameter tuning and log to MLflow.
    Args:
        model_cls: Model class (e.g., RandomForestClassifier)
        model_name (str): Name of the model
        X_train, y_train: Training data
        X_test, y_test: Test data
        params (dict): Model parameters and tuning config
        all_metrics (dict): Dictionary to store metrics
        log_model_func: Function to log model to MLflow
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"🏋️  Training {model_name}")
    logger.info(f"{'='*80}")
    
    with mlflow.start_run(run_name=model_name):
        # Extract base parameters (exclude tuning config)
        base_params = {k: v for k, v in params.items() if k != "tuning"}
        tuning_cfg = params.get("tuning", {})
        
        # Initialize model
        model = model_cls(**base_params)
        
        # Hyperparameter tuning
        if tuning_cfg.get("enabled", False):
            logger.info(f"🔍 Hyperparameter tuning enabled")
            
            search_type = tuning_cfg.get("search_type", "grid")
            param_grid = tuning_cfg["param_grid"]
            cv = tuning_cfg.get("cv", 3)
            
            logger.info(f"   • Search type: {search_type}")
            logger.info(f"   • CV folds: {cv}")
            logger.info(f"   • Param grid: {param_grid}")
            
            if search_type == "grid":
                search = GridSearchCV(
                    model,
                    param_grid,
                    cv=cv,
                    n_jobs=-1,
                    scoring="f1",
                    verbose=1
                )
            else:
                n_iter = tuning_cfg.get("n_iter", 5)
                search = RandomizedSearchCV(
                    model,
                    param_grid,
                    cv=cv,
                    n_iter=n_iter,
                    n_jobs=-1,
                    scoring="f1",
                    random_state=42,
                    verbose=1
                )
            
            logger.info(f"🚀 Starting {search_type} search...")
            search.fit(X_train, y_train)
            
            best_model = search.best_estimator_
            best_params = search.best_params_
            
            logger.info(f"✅ Best parameters found: {best_params}")
            mlflow.log_params(best_params)
        else:
            logger.info(f"⚡ Training with default parameters")
            model.fit(X_train, y_train)
            best_model = model
            mlflow.log_params(base_params)
        
        # Evaluate on test set
        logger.info(f"📊 Evaluating {model_name}...")
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]
        
        metrics = evaluate_model(y_test, y_pred, y_pred_prob)
        all_metrics[model_name] = metrics
        
        # Log metrics
        log_metrics_to_mlflow(metrics)
        
        # Display metrics
        logger.info(f"\n{'='*80}")
        logger.info(f"📈 {model_name} Results:")
        logger.info(f"{'='*80}")
        for metric, value in metrics.items():
            status = "✅" if value >= 0.80 else "⚠️"
            logger.info(f"   {status} {metric.upper():<12}: {value:.4f} ({value*100:.2f}%)")
        logger.info(f"{'='*80}\n")
        
        # Log model as pyfunc
        input_example = X_test.iloc[:2].astype(np.float64)  # ← Ensure correct type
        log_model_func(best_model, input_example)


        # Log preprocessors
        logger.info(f"📦 Logging preprocessors...")
        for file_name in PREPROCESSORS_DIR.iterdir():
            if file_name.is_file():
                mlflow.log_artifact(str(file_name), artifact_path="preprocessors")
        
        logger.info(f"✅ {model_name} training completed!\n")


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    all_metrics: dict
):
    """Train Random Forest model."""

    def log_model_func(model, input_example):
        # Si input_example est un DataFrame
        if hasattr(input_example, "to_dict"):
            input_example = input_example.to_dict(orient="records")

        # Si c’est un ndarray, on le convertit en liste de dicts
        elif hasattr(input_example, "tolist"):
            input_example = [dict(enumerate(row)) for row in input_example.tolist()]

        mlflow.pyfunc.log_model(
            python_model=WrappedModel(model),
            artifact_path="model",
            input_example=input_example
        )   




    
    tune_and_log_model(
        RandomForestClassifier,
        "RandomForest",
        X_train, y_train,
        X_test, y_test,
        config["random_forest"],
        all_metrics,
        log_model_func
    )


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    all_metrics: dict
):
    """Train XGBoost model."""


    def log_model_func(model, input_example):
    # Si input_example est un DataFrame
        if hasattr(input_example, "to_dict"):
            input_example = input_example.to_dict(orient="records")

        # Si c’est un ndarray ou une liste de listes
        elif hasattr(input_example, "tolist"):
            input_example = [dict(enumerate(row)) for row in input_example.tolist()]

        mlflow.pyfunc.log_model(
            python_model=WrappedModel(model),
            artifact_path="model",
            input_example=input_example
        )    

    
    tune_and_log_model(
        xgb.XGBClassifier,
        "XGBoost",
        X_train, y_train,
        X_test, y_test,
        config["xgboost"],
        all_metrics,
        log_model_func
    )




def train_neural_network(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    all_metrics: dict
):
    """Train Neural Network (PyTorch) model."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🏋️  Training Neural Network")
    logger.info(f"{'='*80}")
    
    nn_cfg = config["neural_net"]
    tuning_cfg = nn_cfg.get("tuning", {})
    
    best_metrics = None
    best_params = None
    best_model = None
    
    with mlflow.start_run(run_name="NeuralNet"):
        # Parameter grid
        if tuning_cfg.get("enabled", False):
            param_grid = tuning_cfg.get("param_grid", {})
            logger.info(f"🔍 Hyperparameter tuning enabled")
            logger.info(f"   • Param grid: {param_grid}")
        else:
            param_grid = {
                "hidden_units": [nn_cfg["hidden_units"]],
                "lr": [nn_cfg["lr"]],
                "batch_size": [nn_cfg["batch_size"]]
            }
            logger.info(f"⚡ Training with default parameters")
        
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
        
        # Grid search
        total_configs = len(param_grid["hidden_units"]) * len(param_grid["lr"]) * len(param_grid["batch_size"])
        logger.info(f"🚀 Testing {total_configs} configurations...")
        
        config_num = 0
        for hu in param_grid["hidden_units"]:
            for lr in param_grid["lr"]:
                for bs in param_grid["batch_size"]:
                    config_num += 1
                    logger.info(f"\n--- Config {config_num}/{total_configs}: hidden={hu}, lr={lr}, batch={bs} ---")
                    
                    # Create model
                    input_dim = X_train.shape[1]
                    dropout = nn_cfg.get("dropout", 0.3)
                    model = NN(input_dim, hu, dropout).to(device)
                    
                    # Loss and optimizer
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    
                    # Prepare data
                    X_train_t = torch.from_numpy(X_train.to_numpy(dtype=np.float32))
                    y_train_t = torch.from_numpy(y_train.to_numpy(dtype=np.float32).reshape(-1, 1))
                    X_test_t = torch.from_numpy(X_test.to_numpy(dtype=np.float32))
                    
                    train_loader = DataLoader(
                        TensorDataset(X_train_t, y_train_t),
                        batch_size=bs,
                        shuffle=True
                    )
                    
                    # Training loop
                    epochs = nn_cfg["epochs"]
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
                        
                        if (epoch + 1) % 5 == 0:
                            avg_loss = epoch_loss / len(train_loader)
                            logger.info(f"   Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                    
                    # Evaluate
                    model.eval()
                    with torch.no_grad():
                        logits = model(X_test_t.to(device))
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                        preds = (probs > 0.5).astype(int)
                    
                    metrics = evaluate_model(y_test, preds, probs)
                    
                    # Display metrics
                    logger.info(f"   📊 Metrics:")
                    for metric, value in metrics.items():
                        status = "✅" if value >= 0.80 else "⚠️"
                        logger.info(f"      {status} {metric.upper():<12}: {value:.4f}")
                    
                    # Keep best model
                    if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
                        best_metrics = metrics
                        best_params = {"hidden_units": hu, "lr": lr, "batch_size": bs}
                        best_model = model.to("cpu")  # Move to CPU for saving
                        logger.info(f"   ⭐ New best model!")
        
        # Log best results
        logger.info(f"\n{'='*80}")
        logger.info(f"📈 Neural Network Best Results:")
        logger.info(f"{'='*80}")
        logger.info(f"Best params: {best_params}")
        for metric, value in best_metrics.items():
            status = "✅" if value >= 0.80 else "⚠️"
            logger.info(f"   {status} {metric.upper():<12}: {value:.4f} ({value*100:.2f}%)")
        logger.info(f"{'='*80}\n")
        
        mlflow.log_params(best_params)
        log_metrics_to_mlflow(best_metrics)
        all_metrics["NeuralNet"] = best_metrics
        

        # Log model
        input_example = X_test.iloc[:2].to_numpy().astype(np.float64).tolist()  # ← .tolist() à la fin
        mlflow.pyfunc.log_model(
            python_model=PyTorchWrapper(best_model),
            artifact_path="model",
            input_example=input_example
        )

        
        # Log preprocessors
        for file_name in PREPROCESSORS_DIR.iterdir():
            if file_name.is_file():
                mlflow.log_artifact(str(file_name), artifact_path="preprocessors")
        
        logger.info(f"✅ Neural Network training completed!\n")


def train_all_models(df: pd.DataFrame):
    """
    Train all models with SMOTE and log to MLflow.
    
    Args:
        df (pd.DataFrame): Feature-engineered dataframe
    """
    logger.info("\n" + "="*80)
    logger.info("🚀 STARTING MODEL TRAINING PIPELINE")
    logger.info("="*80 + "\n")
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info(f"📊 MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"🧪 MLflow experiment: {MLFLOW_EXPERIMENT_NAME}\n")
    
    all_metrics = {}
    
    # Split data
    train_cfg = config["training"]
    X_train, X_test, y_train, y_test = split_data(
        df,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"]
    )
    
    # Apply SMOTE if enabled
    if train_cfg.get("use_smote", True):
        logger.info("\n" + "="*80)
        logger.info("🔄 Applying SMOTE to balance training data")
        logger.info("="*80)
        
        logger.info(f"   • Before SMOTE:")
        logger.info(f"     - Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
        logger.info(f"     - Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
        
        smote_strategy = train_cfg.get("smote_strategy", 0.6)  # Default 60% minority
        smote = SMOTE(
            sampling_strategy=smote_strategy,
            random_state=train_cfg.get("smote_random_state", 2)
        )

        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        logger.info(f"   • After SMOTE:")
        logger.info(f"     - Class 0: {(y_train == 0).sum()} ({(y_train == 0).sum() / len(y_train) * 100:.1f}%)")
        logger.info(f"     - Class 1: {(y_train == 1).sum()} ({(y_train == 1).sum() / len(y_train) * 100:.1f}%)")
        logger.info(f"   • SMOTE strategy: {smote_strategy} (target minority ratio)")
        logger.info(f"✅ SMOTE completed: {len(y_train)} samples\n")
    
    # Train models
    train_random_forest(X_train, y_train, X_test, y_test, all_metrics)
    train_xgboost(X_train, y_train, X_test, y_test, all_metrics)
    train_neural_network(X_train, y_train, X_test, y_test, all_metrics)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 FINAL RESULTS SUMMARY")
    logger.info("="*80 + "\n")
    
    # Find best model
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["f1"])
    
    for model_name, metrics in all_metrics.items():
        is_best = "🏆" if model_name == best_model_name else "  "
        logger.info(f"{is_best} {model_name}:")
        for metric, value in metrics.items():
            status = "✅" if value >= 0.80 else "⚠️"
            logger.info(f"      {status} {metric.upper():<12}: {value:.4f} ({value*100:.2f}%)")
        logger.info("")
    
    logger.info(f"🏆 Best model: {best_model_name} (F1={all_metrics[best_model_name]['f1']:.4f})")
    
    # Check if all metrics > 80% for best model
    best_metrics = all_metrics[best_model_name]
    all_above_80 = all(v >= 0.80 for v in best_metrics.values())
    
    if all_above_80:
        logger.info(f"\n🎉 SUCCESS! All metrics above 80% for {best_model_name}! 🎉")
    else:
        logger.info(f"\n⚠️  Some metrics below 80%. Consider:")
        logger.info(f"   • More feature engineering")
        logger.info(f"   • Different hyperparameters")
        logger.info(f"   • Ensemble methods")
    
    logger.info("\n" + "="*80)
    logger.info("✅ MODEL TRAINING COMPLETED")
    logger.info("="*80 + "\n")
    
    logger.info(f"💡 View results in MLflow UI:")
    logger.info(f"   mlflow ui")
    logger.info(f"   Then open: http://localhost:5000\n")


def register_best_model():
    """
    Register the best model (highest F1 score) to MLflow Model Registry.
    """
    logger.info("\n" + "="*80)
    logger.info("📦 REGISTERING BEST MODEL")
    logger.info("="*80 + "\n")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    
    if experiment is None:
        logger.error(f"❌ Experiment '{MLFLOW_EXPERIMENT_NAME}' not found")
        return None
    
    # Find best run by F1 score
    runs = client.search_runs(
        experiment.experiment_id,
        order_by=["metrics.f1 DESC"],
        max_results=1
    )
    
    if not runs:
        logger.warning(f"⚠️  No runs found for experiment '{MLFLOW_EXPERIMENT_NAME}'")
        return None
    
    best_run = runs[0]
    
    logger.info(f"🏆 Best run found:")
    logger.info(f"   • Run ID: {best_run.info.run_id}")
    logger.info(f"   • Run name: {best_run.data.tags.get('mlflow.runName', 'Unknown')}")
    logger.info(f"   • F1 score: {best_run.data.metrics['f1']:.4f}")
    logger.info(f"   • Accuracy: {best_run.data.metrics.get('accuracy', 0):.4f}")
    logger.info(f"   • ROC AUC: {best_run.data.metrics.get('roc_auc', 0):.4f}")
    
    # Register model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    
    try:
        registered_model = mlflow.register_model(model_uri, MLFLOW_EXPERIMENT_NAME)
        logger.info(f"\n✅ Model registered successfully!")
        logger.info(f"   • Name: {registered_model.name}")
        logger.info(f"   • Version: {registered_model.version}")
        logger.info(f"\n💡 Promote to production:")
        logger.info(f"   mlflow models transition --name {MLFLOW_EXPERIMENT_NAME} --version {registered_model.version} --stage Production")
    
    except Exception as e:
        logger.error(f"❌ Error registering model: {e}")
        raise
    
    logger.info("\n" + "="*80)
    logger.info("✅ MODEL REGISTRATION COMPLETED")
    logger.info("="*80 + "\n")


def main():
    """Main training pipeline."""
    try:
        # 
        # Use advanced features file (50 features instead of 13)
        feature_path = FEATURE_DATA_DIR / "telco_churn_features_advanced.csv"
        
        if not feature_path.exists():
            logger.error(f"❌ Feature data not found at: {feature_path}")
            logger.error(f"   Please run feature engineering first!")
            raise FileNotFoundError(f"Feature data not found: {feature_path}")
        
        logger.info(f"📂 Loading feature data from: {feature_path}")
        df = pd.read_csv(feature_path)
        logger.info(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns\n")
        
        # Train all models
        train_all_models(df)
        
        # Register best model
        register_best_model()
        
        logger.info("🎉 All done! Check MLflow UI for detailed results.")
    
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()