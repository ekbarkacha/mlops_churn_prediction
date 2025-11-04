"""
Pytest Configuration & Shared Fixtures
Professional-grade test fixtures for ML pipeline
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil
from typing import Dict, Tuple
import yaml

# Add src to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.const import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_DATA_DIR,
    PREPROCESSORS_DIR,
    MODEL_CONFIG_PATH
)


# ============================================================================
# SESSION FIXTURES (Setup once per test session)
# ============================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="test_churn_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def model_config():
    """Load model configuration"""
    with open(MODEL_CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ============================================================================
# SAMPLE DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_raw_data() -> pd.DataFrame:
    """
    Sample raw data mimicking Telco Churn dataset
    Small, fast, perfect for unit tests
    """
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'customerID': [f'C{i:04d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(0, 73, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice([
            'Electronic check', 
            'Mailed check', 
            'Bank transfer (automatic)',
            'Credit card (automatic)'
        ], n_samples),
        'MonthlyCharges': np.random.uniform(18.0, 120.0, n_samples).round(2),
        'TotalCharges': [
            str(round(np.random.uniform(18.0, 8000.0), 2)) if np.random.random() > 0.1 else ' '
            for _ in range(n_samples)
        ],
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    })


@pytest.fixture
def sample_processed_data() -> pd.DataFrame:
    """
    Sample preprocessed data (after cleaning & encoding)
    """
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice([0, 1], n_samples),
        'Dependents': np.random.choice([0, 1], n_samples),
        'tenure': np.random.uniform(0, 1, n_samples),  # Scaled
        'OnlineSecurity': np.random.choice([0, 1, 2], n_samples),
        'OnlineBackup': np.random.choice([0, 1, 2], n_samples),
        'DeviceProtection': np.random.choice([0, 1, 2], n_samples),
        'TechSupport': np.random.choice([0, 1, 2], n_samples),
        'Contract': np.random.choice([0, 1, 2], n_samples),
        'PaperlessBilling': np.random.choice([0, 1], n_samples),
        'PaymentMethod': np.random.choice([0, 1, 2, 3], n_samples),
        'MonthlyCharges': np.random.uniform(0, 1, n_samples),  # Scaled
        'TotalCharges': np.random.uniform(0, 1, n_samples),  # Scaled
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    })


@pytest.fixture
def sample_features_data() -> pd.DataFrame:
    """
    Sample feature-engineered data (42 features)
    Includes advanced features
    """
    np.random.seed(42)
    n_samples = 100
    
    # Base features
    base_features = {
        'SeniorCitizen': np.random.choice([0, 1], n_samples),
        'Partner': np.random.choice([0, 1], n_samples),
        'Dependents': np.random.choice([0, 1], n_samples),
        'tenure': np.random.uniform(0, 1, n_samples),
        'OnlineSecurity': np.random.choice([0, 1, 2], n_samples),
        'OnlineBackup': np.random.choice([0, 1, 2], n_samples),
        'DeviceProtection': np.random.choice([0, 1, 2], n_samples),
        'TechSupport': np.random.choice([0, 1, 2], n_samples),
        'Contract': np.random.choice([0, 1, 2], n_samples),
        'PaperlessBilling': np.random.choice([0, 1], n_samples),
        'PaymentMethod': np.random.choice([0, 1, 2, 3], n_samples),
        'MonthlyCharges': np.random.uniform(0, 1, n_samples),
        'TotalCharges': np.random.uniform(0, 1, n_samples),
    }
    
    df = pd.DataFrame(base_features)
    
    # Advanced features (simulated)
    df['tenure_group'] = pd.cut(df['tenure'] * 72, bins=[0, 12, 24, 48, 72], labels=[0, 1, 2, 3])
    df['tenure_group'] = df['tenure_group'].astype(int)
    
    df['charges_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1e-6)
    df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1e-6)
    
    # Interaction features
    df['tenure_contract_interaction'] = df['tenure'] * df['Contract']
    df['tenure_charges_product'] = df['tenure'] * df['MonthlyCharges']
    
    # Service features
    df['total_services'] = (
        (df['OnlineSecurity'] > 0).astype(int) +
        (df['OnlineBackup'] > 0).astype(int) +
        (df['DeviceProtection'] > 0).astype(int) +
        (df['TechSupport'] > 0).astype(int)
    )
    
    # Add more features to reach 42
    for i in range(42 - len(df.columns) - 1):  # -1 for Churn
        df[f'feature_{i}'] = np.random.uniform(0, 1, n_samples)
    
    # Target
    df['Churn'] = np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    
    return df


@pytest.fixture
def train_test_split_data(sample_features_data) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Pre-split data for training tests
    Returns: X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    df = sample_features_data
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# PERFORMANCE THRESHOLDS (Critical for validation)
# ============================================================================

@pytest.fixture
def model_performance_thresholds() -> Dict[str, float]:
    """
    Minimum acceptable model performance thresholds
    Based on production requirements
    """
    return {
        'f1': 0.60,          # F1 score >= 60%
        'roc_auc': 0.80,     # ROC AUC >= 80%
        'precision': 0.50,   # Precision >= 50%
        'recall': 0.70,      # Recall >= 70%
        'accuracy': 0.70     # Accuracy >= 70%
    }


@pytest.fixture
def model_performance_targets() -> Dict[str, float]:
    """
    Target performance (what we achieved in production)
    """
    return {
        'f1': 0.6403,
        'roc_auc': 0.8467,
        'precision': 0.5155,
        'recall': 0.8449,
        'accuracy': 0.7480
    }


# ============================================================================
# DATA VALIDATION SCHEMAS
# ============================================================================

@pytest.fixture
def raw_data_schema() -> Dict:
    """Expected schema for raw data"""
    return {
        'required_columns': [
            'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'
        ],
        'numeric_columns': ['SeniorCitizen', 'tenure', 'MonthlyCharges'],
        'categorical_columns': [
            'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'Churn'
        ],
        'target_column': 'Churn',
        'min_rows': 1000  # Minimum expected rows
    }


@pytest.fixture
def processed_data_schema() -> Dict:
    """Expected schema for processed data"""
    return {
        'required_columns': [
            'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges', 'Churn'
        ],
        'numeric_only': True,  # All columns should be numeric
        'no_missing': True,    # No missing values allowed
        'target_column': 'Churn',
        'target_values': [0, 1]
    }


# ============================================================================
# HELPER FUNCTIONS (Available in all tests)
# ============================================================================

@pytest.fixture
def assert_dataframe_schema():
    """Helper to validate dataframe schema"""
    def _validate(df: pd.DataFrame, schema: Dict):
        # Check required columns
        if 'required_columns' in schema:
            missing = set(schema['required_columns']) - set(df.columns)
            assert not missing, f"Missing columns: {missing}"
        
        # Check no missing values
        if schema.get('no_missing', False):
            assert df.isnull().sum().sum() == 0, "Found missing values"
        
        # Check numeric only
        if schema.get('numeric_only', False):
            non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
            if 'customerID' in non_numeric:
                non_numeric.remove('customerID')
            assert not non_numeric, f"Non-numeric columns found: {non_numeric}"
        
        # Check target values
        if 'target_column' in schema and 'target_values' in schema:
            target_col = schema['target_column']
            if target_col in df.columns:
                unique_vals = set(df[target_col].unique())
                expected_vals = set(schema['target_values'])
                assert unique_vals.issubset(expected_vals), \
                    f"Unexpected target values: {unique_vals - expected_vals}"
        
        return True
    
    return _validate


@pytest.fixture
def assert_model_performance():
    """Helper to validate model performance against thresholds"""
    def _validate(metrics: Dict[str, float], thresholds: Dict[str, float]):
        failures = []
        for metric, threshold in thresholds.items():
            if metric in metrics:
                if metrics[metric] < threshold:
                    failures.append(
                        f"{metric}: {metrics[metric]:.4f} < {threshold:.4f}"
                    )
        
        assert not failures, f"Performance below thresholds:\n" + "\n".join(failures)
        return True
    
    return _validate


# ============================================================================
# PYTEST HOOKS (Advanced)
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests (fast, isolated)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (slower)"
    )
    config.addinivalue_line(
        "markers", "validation: Model validation tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (training, large data)"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on path"""
    for item in items:
        # Auto-mark based on directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "validation" in str(item.fspath):
            item.add_marker(pytest.mark.validation)
        
        # Mark slow tests
        if "train" in item.name.lower() or "pipeline" in item.name.lower():
            item.add_marker(pytest.mark.slow)