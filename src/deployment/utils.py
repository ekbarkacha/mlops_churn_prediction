"""
Deployment Utilities - Model Loading & Validation
Compatible with Python 3.11.6
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Any, Optional
import mlflow
import logging
from pathlib import Path

from .config import config

logger = logging.getLogger(__name__)


class ModelLoader:
    """
    Singleton Model Loader with lazy initialization
    Thread-safe model loading and caching
    """
    
    _instance: Optional['ModelLoader'] = None
    _model: Optional[Any] = None
    _model_metadata: Optional[Dict] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_model(self, force_reload: bool = False) -> Any:
        """
        Load production model from MLflow
        
        Args:
            force_reload: Force reload even if cached
        
        Returns:
            Loaded model
        
        Raises:
            ValueError: If model cannot be loaded
        """
        if self._model is not None and not force_reload:
            logger.debug("Using cached model")
            return self._model
        
        logger.info("Loading production model from MLflow...")
        
        try:
            # Set MLflow tracking
            mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
            
            # Get experiment
            client = mlflow.tracking.MlflowClient()

                 # === MODIFIER ICI ===
        # Chercher l'experiment par nom
            experiment = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
        
            if experiment is None:
                # Fallback: chercher tous les experiments et trouver "churn_prediction"
                logger.warning(f"Experiment '{config.MLFLOW_EXPERIMENT_NAME}' not found by name")
                all_experiments = client.search_experiments()
                
                for exp in all_experiments:
                    if exp.name == config.MLFLOW_EXPERIMENT_NAME:
                        experiment = exp
                        logger.info(f"✅ Found experiment by search: {exp.name} (ID: {exp.experiment_id})")
                        break
                
                if experiment is None:
                    raise ValueError(f"Experiment '{config.MLFLOW_EXPERIMENT_NAME}' not found")
        
            logger.info(f"🔍 Using experiment: {experiment.name} (ID: {experiment.experiment_id})")
        # 
            
            
            
            # Get best run (highest F1)
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.f1 DESC"],
                max_results=1
            )
            
            if not runs:
                raise ValueError("No runs found in experiment")
            
            best_run = runs[0]
            run_id = best_run.info.run_id
            
            # Load model
            model_uri = f"runs:/{run_id}/model"
            self._model = mlflow.pyfunc.load_model(model_uri)
            
            # Store metadata
            self._model_metadata = {
                'run_id': run_id,
                'run_name': best_run.data.tags.get('mlflow.runName', 'Unknown'),
                'f1_score': best_run.data.metrics.get('f1', 0.0),
                'roc_auc': best_run.data.metrics.get('roc_auc', 0.0),
                'accuracy': best_run.data.metrics.get('accuracy', 0.0)
            }
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Run ID: {run_id[:8]}...")
            logger.info(f"   F1: {self._model_metadata['f1_score']:.4f}")
            logger.info(f"   ROC AUC: {self._model_metadata['roc_auc']:.4f}")
            
            return self._model
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise ValueError(f"Model loading failed: {e}")
    
    def get_model_info(self) -> Dict:
        """Get model metadata"""
        if self._model_metadata is None:
            self.load_model()
        return self._model_metadata or {}
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model is not None


# Singleton instance
model_loader = ModelLoader()


def validate_input_data(data: Union[Dict, List[Dict]]) -> pd.DataFrame:
    """
    Validate and convert input to DataFrame
    
    Args:
        data: Input data (dict or list of dicts)
    
    Returns:
        Validated DataFrame
    
    Raises:
        ValueError: If data is invalid
    """
    try:
        # Convert to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            if not data:
                raise ValueError("Data list cannot be empty")
            df = pd.DataFrame(data)
        else:
            raise ValueError("Data must be dict or list of dicts")
        
        # Check not empty
        if df.empty:
            raise ValueError("Data cannot be empty")
        
        # Check size limit
        if len(df) > config.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum ({config.MAX_BATCH_SIZE})")
        
        logger.debug(f"Input validated: {df.shape}")
        return df
        
    except Exception as e:
        raise ValueError(f"Invalid input data: {e}")



def format_prediction_response(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    input_size: int
) -> List[Dict[str, Any]]:
    """Format predictions into response structure"""
    

    
    results = []
    for i in range(input_size):
        prob = float(probabilities[i])
        pred = int(predictions[i])
        
        # Determine risk level based on probability
        if prob >= 0.7:
            risk_level = "HIGH"
        elif prob >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        results.append({
            "prediction": "Churn" if pred == 1 else "No Churn",
            "churn_probability": round(prob, 4),  # Probabilité de churn
            "confidence": round(max(prob, 1 - prob), 4),  # Confiance = max(prob, 1-prob)
            "risk_level": risk_level
        })
    
    return results




def get_batch_stats(predictions: np.ndarray) -> Dict[str, Any]:
    """
    Calculate batch prediction statistics
    
    Args:
        predictions: Array of predictions
    
    Returns:
        Statistics dictionary
    """
    return {
        'total_predictions': int(len(predictions)),
        'churn_predicted': int(np.sum(predictions)),
        'no_churn_predicted': int(len(predictions) - np.sum(predictions)),
        'churn_rate': float(np.mean(predictions))
    }
