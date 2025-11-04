"""
Smoke test - Vérifie que tout fonctionne
"""
import pytest


def test_pytest_works():
    """Test basique pour vérifier que pytest fonctionne"""
    assert True


def test_fixtures_available(sample_raw_data, model_performance_thresholds):
    """Test que les fixtures sont accessibles"""
    assert sample_raw_data is not None
    assert len(sample_raw_data) > 0
    assert 'Churn' in sample_raw_data.columns
    
    assert model_performance_thresholds is not None
    assert 'f1' in model_performance_thresholds
    assert model_performance_thresholds['f1'] > 0


@pytest.mark.unit
def test_sample_data_schema(sample_raw_data, raw_data_schema):
    """Test que les données de test ont le bon schéma"""
    required_cols = raw_data_schema['required_columns']
    assert all(col in sample_raw_data.columns for col in required_cols)


def test_numpy_pandas_available():
    """Test que les dépendances critiques sont installées"""
    import numpy as np
    import pandas as pd
    
    arr = np.array([1, 2, 3])
    df = pd.DataFrame({'a': [1, 2, 3]})
    
    assert len(arr) == 3
    assert len(df) == 3