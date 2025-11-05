"""
Script de Diagnostic d'Environnement
Vérifie toutes les dépendances et configurations du projet
"""
import sys
import importlib
import platform
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def check_python_version():
    """Vérifie la version de Python"""
    version = sys.version_info
    print(f"\n{'='*70}")
    print("[Python] VERSION PYTHON")
    print(f"{'='*70}")
    print(f"Version: {version.major}.{version.minor}.{version.micro}")
    print(f"Plateforme: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")

    if version.major != 3 or version.minor != 11:
        print("[!] ATTENTION: Python 3.11 est recommandé")
        return False
    else:
        print("[OK] Version Python compatible")
        return True

def check_packages():
    """Vérifie tous les packages requis"""
    print(f"\n{'='*70}")
    print("[Packages] PACKAGES INSTALLES")
    print(f"{'='*70}")

    required_packages = {
        # Core ML
        'numpy': '1.26.3',
        'pandas': '2.2.0',
        'sklearn': '1.4.0',
        'scipy': '1.16.3',
        'imblearn': '0.12.0',

        # Deep Learning
        'torch': '2.2.0',
        'xgboost': '2.0.3',

        # MLOps
        'mlflow': '2.9.2',

        # Visualization
        'matplotlib': '3.10.7',
        'seaborn': '0.13.2',
        'plotly': '6.4.0',

        # API & Deployment
        'fastapi': '0.109.0',
        'uvicorn': '0.27.0',
        'pydantic': '2.5.3',

        # Dashboard
        'streamlit': '1.51.0',

        # Testing
        'pytest': '8.0.0',

        # Utilities
        'yaml': None,  # PyYAML
        'colorlog': '6.8.2',
        'kaggle': '1.6.6',
    }

    installed = []
    missing = []
    version_mismatch = []

    for package, expected_version in required_packages.items():
        try:
            # Import the module
            if package == 'sklearn':
                mod = importlib.import_module('sklearn')
            elif package == 'imblearn':
                mod = importlib.import_module('imblearn')
            elif package == 'yaml':
                mod = importlib.import_module('yaml')
            else:
                mod = importlib.import_module(package)

            # Get version
            version = getattr(mod, '__version__', 'N/A')

            # Check version match
            if expected_version and version != expected_version and not version.startswith(expected_version.split('.')[0]):
                version_mismatch.append((package, expected_version, version))
                print(f"[!] {package:20s} : {version} (attendu: {expected_version})")
            else:
                installed.append(package)
                print(f"[OK] {package:20s} : {version}")

        except ImportError:
            missing.append(package)
            print(f"[X] {package:20s} : MANQUANT")

    return installed, missing, version_mismatch

def check_project_structure():
    """Vérifie la structure du projet"""
    print(f"\n{'='*70}")
    print("[Structure] STRUCTURE DU PROJET")
    print(f"{'='*70}")

    required_dirs = [
        'src/data_pipeline',
        'src/modeling',
        'src/deployment',
        'src/dashboard',
        'src/utils',
        'tests',
        'data',
        'config',
        'artifacts',
    ]

    required_files = [
        'main.py',
        'requirements.txt',
        'pytest.ini',
        'config/model_config.yaml',
        'src/utils/const.py',
        'src/utils/logger.py',
    ]

    all_ok = True

    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"[OK] {dir_path:40s} : EXISTS")
        else:
            print(f"[X] {dir_path:40s} : MANQUANT")
            all_ok = False

    print()
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"[OK] {file_path:40s} : EXISTS")
        else:
            print(f"[X] {file_path:40s} : MANQUANT")
            all_ok = False

    return all_ok

def check_data_files():
    """Vérifie les fichiers de données"""
    print(f"\n{'='*70}")
    print("[Data] DONNEES")
    print(f"{'='*70}")

    data_files = [
        'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv',
        'data/processed/telco_churn_processed.csv',
        'data/features/telco_churn_features.csv',
    ]

    for file_path in data_files:
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"[OK] {file_path:50s} : {size_mb:.2f} MB")
        else:
            print(f"[~] {file_path:50s} : NON GENERE")

def check_mlflow():
    """Vérifie l'installation MLflow"""
    print(f"\n{'='*70}")
    print("[MLflow] MLFLOW")
    print(f"{'='*70}")

    mlruns_path = Path("mlruns")
    if mlruns_path.exists():
        experiments = list(mlruns_path.glob("*/"))
        print(f"[OK] MLflow tracking directory exists")
        print(f"   Experiments: {len(experiments)}")

        # Count runs
        total_runs = 0
        for exp in experiments:
            if exp.name != '.trash':
                runs = list(exp.glob("*/"))
                total_runs += len(runs)
        print(f"   Total runs: {total_runs}")
    else:
        print(f"[~] MLflow tracking directory not initialized")

def generate_recommendations(missing_packages, version_mismatches, structure_ok):
    """Génère des recommandations"""
    print(f"\n{'='*70}")
    print("[Recommendations] RECOMMANDATIONS")
    print(f"{'='*70}")

    if not missing_packages and not version_mismatches and structure_ok:
        print("\n[SUCCESS] TOUT EST PARFAIT!")
        print("Votre environnement est pret pour le developpement.")
        print("\nProchaines etapes:")
        print("   1. python main.py                          # Executer le pipeline")
        print("   2. uvicorn src.deployment.api:app --reload # Lancer l'API")
        print("   3. streamlit run src/dashboard/app.py      # Lancer le dashboard")
        print("   4. pytest tests/ -v                        # Executer les tests")
        return

    if missing_packages:
        print(f"\n[!] {len(missing_packages)} packages manquants:")
        print("   Action: Executez install_all.bat ou:")
        print("   pip install -r requirements.txt")

    if version_mismatches:
        print(f"\n[!] {len(version_mismatches)} packages avec versions incorrectes:")
        for pkg, expected, actual in version_mismatches:
            print(f"      {pkg}: {actual} -> {expected}")
        print("   Action: Reinstallez les packages:")
        print("   pip install -r requirements.txt --upgrade")

    if not structure_ok:
        print("\n[!] Structure du projet incomplete")
        print("   Action: Assurez-vous d'etre dans le bon repertoire")
        print("   cd C:\\Users\\pc\\Desktop\\mlops_churn_prediction")

def main():
    """Fonction principale"""
    print("\n" + "="*70)
    print("[Diagnostic] DIAGNOSTIC D'ENVIRONNEMENT - MLOps Churn Prediction")
    print("="*70)

    # Checks
    python_ok = check_python_version()
    installed, missing, mismatches = check_packages()
    structure_ok = check_project_structure()
    check_data_files()
    check_mlflow()

    # Summary
    print(f"\n{'='*70}")
    print("[Summary] RESUME")
    print(f"{'='*70}")
    print(f"[OK] Packages installes: {len(installed)}")
    print(f"[X] Packages manquants: {len(missing)}")
    print(f"[!] Versions incorrectes: {len(mismatches)}")
    print(f"[{'OK' if structure_ok else 'X'}] Structure du projet: {'OK' if structure_ok else 'PROBLEMES'}")

    # Recommendations
    generate_recommendations(missing, mismatches, structure_ok)

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
