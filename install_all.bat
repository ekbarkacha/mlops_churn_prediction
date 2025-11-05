@echo off
REM ============================================================================
REM Script d'Installation Automatique - MLOps Churn Prediction
REM Python 3.11.6 - Windows
REM ============================================================================

echo.
echo ============================================================================
echo         INSTALLATION MLOps CHURN PREDICTION
echo ============================================================================
echo.

REM Check Python version
echo [1/5] Verification de Python...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python n'est pas installe ou pas dans le PATH
    pause
    exit /b 1
)
echo.

REM Upgrade pip
echo [2/5] Mise a jour de pip...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo ERREUR: Impossible de mettre a jour pip
    pause
    exit /b 1
)
echo.

REM Install PyTorch
echo [3/5] Installation de PyTorch (CPU version)...
echo ATTENTION: Cela peut prendre quelques minutes...
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
if %errorlevel% neq 0 (
    echo ERREUR: Impossible d'installer PyTorch
    pause
    exit /b 1
)
echo.

REM Install critical packages first
echo [4/5] Installation des packages critiques...
pip install pydantic==2.5.3 pydantic-core==2.14.6 pydantic-settings==2.1.0 protobuf==4.25.8
if %errorlevel% neq 0 (
    echo ERREUR: Impossible d'installer les packages critiques
    pause
    exit /b 1
)
echo.

REM Install all other packages
echo [5/5] Installation de toutes les dependances...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERREUR: Impossible d'installer toutes les dependances
    pause
    exit /b 1
)
echo.

REM Verify installation
echo ============================================================================
echo         VERIFICATION DE L'INSTALLATION
echo ============================================================================
echo.

python -c "import torch; print('✓ PyTorch:', torch.__version__)"
python -c "import pandas; print('✓ pandas:', pandas.__version__)"
python -c "import numpy; print('✓ numpy:', numpy.__version__)"
python -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)"
python -c "import mlflow; print('✓ MLflow:', mlflow.__version__)"
python -c "import fastapi; print('✓ FastAPI:', fastapi.__version__)"
python -c "import streamlit; print('✓ Streamlit:', streamlit.__version__)"
python -c "import xgboost; print('✓ XGBoost:', xgboost.__version__)"

echo.
echo ============================================================================
echo         INSTALLATION TERMINEE AVEC SUCCES!
echo ============================================================================
echo.
echo Prochaines etapes:
echo   1. Executer le pipeline: python main.py
echo   2. Lancer l'API: uvicorn src.deployment.api:app --reload
echo   3. Lancer le dashboard: streamlit run src/dashboard/app.py
echo   4. Executer les tests: pytest tests/ -v
echo.
pause
