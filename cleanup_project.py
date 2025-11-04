"""
Project Cleanup Script - Remove unnecessary files and organize for production
"""
#
from pathlib import Path
import shutil
from datetime import datetime

############################################################

def cleanup_project():
    """Clean up project for production deployment"""
    
    print("🧹 STARTING PROJECT CLEANUP")
    print("="*80)
    
    project_root = Path(__file__).parent
    
    # Create archive directory for old experiments
    archive_dir = project_root / "archive"
    archive_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========================================
    # 1. Archive experimental scripts
    # ========================================
    print("\n📦 Archiving experimental scripts...")
    
    experimental_files = [
        "src/modeling/optimize_hyperparameters.py",  # Experiments done
        "src/modeling/feature_selection.py",          # Not used in final
        "src/data_pipeline/advanced_feature_engineering.py"  # Using features_advanced.csv directly
    ]
    
    for file_path in experimental_files:
        source = project_root / file_path
        if source.exists():
            dest = archive_dir / f"{timestamp}_{source.name}"
            shutil.move(str(source), str(dest))
            print(f"   ✅ Archived: {file_path}")
    
    # ========================================
    # 2. Clean up data directories
    # ========================================
    print("\n🗑️  Cleaning data directories...")
    
    # Keep only essential data files
    data_dir = project_root / "data"
    
    # Remove intermediate files (keep only final)
    files_to_keep = [
        "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "data/processed/telco_churn_processed.csv",
        "data/features/telco_churn_features_advanced.csv"
    ]
    
    # Remove feature selection results (not used)
    feature_selected = data_dir / "features" / "telco_churn_features_selected.csv"
    if feature_selected.exists():
        feature_selected.unlink()
        print(f"   ✅ Removed: telco_churn_features_selected.csv")
    
    feature_list = data_dir / "features" / "selected_features_list.txt"
    if feature_list.exists():
        feature_list.unlink()
        print(f"   ✅ Removed: selected_features_list.txt")
    
    # ========================================
    # 3. Clean MLflow runs (keep only best)
    # ========================================
    print("\n📊 MLflow runs kept (manually review in UI)...")
    print("   ℹ️  Open MLflow UI to delete old experimental runs")
    print("   ℹ️  Keep only: Baseline NN (F1=64.51%)")
    
    # ========================================
    # 4. Remove __pycache__ directories
    # ========================================
    print("\n🗑️  Removing cache directories...")
    
    for pycache in project_root.rglob("__pycache__"):
        shutil.rmtree(pycache)
        print(f"   ✅ Removed: {pycache.relative_to(project_root)}")
    
    for pytest_cache in project_root.rglob(".pytest_cache"):
        shutil.rmtree(pytest_cache)
        print(f"   ✅ Removed: {pytest_cache.relative_to(project_root)}")
    
    # ========================================
    # 5. Summary
    # ========================================
    print("\n" + "="*80)
    print("✅ CLEANUP COMPLETED!")
    print("="*80)
    
    print("\n📁 Project Structure (Production-Ready):")
    print("""
    customer-churn-prediction/
    ├── data/
    │   ├── raw/                    ✅ Original data
    │   ├── processed/              ✅ Cleaned data
    │   └── features/               ✅ Final features (42 features)
    ├── src/
    │   ├── data_pipeline/          ✅ ETL pipeline
    │   │   ├── data_ingestion.py
    │   │   ├── data_preprocessing.py
    │   │   └── feature_engineering.py
    │   ├── modeling/               ✅ Model training
    │   │   ├── model_training.py
    │   │   ├── model_utils.py
    │   │   ├── nn_model.py
    │   │   └── retrain_best_model.py  ✅ Production model
    │   └── utils/                  ✅ Utilities
    │       ├── logger.py
    │       ├── const.py
    │       └── config.py
    ├── tests/                      ✅ Unit tests
    ├── config/
    │   └── model_config.yaml       ✅ Final config
    ├── artifacts/
    │   ├── models/                 ✅ Saved models
    │   └── preprocessors/          ✅ Encoders & scalers
    ├── mlruns/                     ✅ MLflow tracking
    ├── archive/                    📦 Old experiments
    ├── requirements.txt            ✅ Dependencies
    └── main.py                     ✅ Main pipeline
    """)
    
    print("\n💡 Next steps:")
    print("   1. Review archived files in ./archive/")
    print("   2. Clean MLflow runs (keep only best model)")
    print("   3. Run production model: python src/modeling/retrain_best_model.py")
    print("   4. Ready for deployment! 🚀\n")


if __name__ == "__main__":
    cleanup_project()