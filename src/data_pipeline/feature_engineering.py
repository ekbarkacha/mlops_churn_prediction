"""
Feature Engineering Module - Select, transform, and scale features
"""
import sys
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import get_logger
from src.utils.const import (
    PROCESSED_DATA_DIR,
    PROCESSED_FILE_NAME,
    FEATURE_DATA_DIR,
    FEATURE_FILE_NAME,
    PREPROCESSORS_DIR,
    SCALER_FILE,
    FEATURES_TO_DROP,
    FEATURES_TO_SCALE,
    TARGET_COLUMN,
    FINAL_FEATURES
)

# Initialize logger
logger = get_logger(__name__)


def feature_creation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    
    Currently a placeholder - no new features created yet.
    Future ideas:
    - tenure_per_charge = tenure / (MonthlyCharges + 1)
    - is_new_customer = tenure < 6
    - total_services = sum of all services
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    logger.info("🔧 Feature creation (currently no new features)...")
    
    # Placeholder - no features created yet
    # Based on the analyzed project, they kept it simple
    
    logger.info("✅ Feature creation completed (no changes)")
    return df


def feature_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transformations to features.
    
    Currently a placeholder - no transformations applied.
    Future ideas:
    - Log transformation for skewed features
    - Polynomial features
    - Feature interactions
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Transformed dataframe
    """
    logger.info("🔄 Feature transformation (currently no transformations)...")
    
    # Placeholder - no transformations yet
    # The analyzed project kept it simple: raw features work well
    
    logger.info("✅ Feature transformation completed (no changes)")
    return df


def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features by dropping non-predictive/redundant columns.
    
    Drops 7 features:
    - customerID: Unique identifier, no predictive value
    - PhoneService: Redundant with MultipleLines
    - gender: Low correlation with Churn
    - StreamingTV, StreamingMovies: Optional services, low impact
    - MultipleLines: Correlated with other telecom features
    - InternetService: Redundant, captured by other features
    
    Args:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe with selected features
    
    Raises:
        RuntimeError: If feature selection fails
    """
    logger.info("✂️  Starting feature selection...")
    
    try:
        df = df.copy(deep=True)
        
        # Log initial state
        initial_cols = df.columns.tolist()
        logger.info(f"   • Initial features: {len(initial_cols)}")
        logger.info(f"   • Features to drop: {FEATURES_TO_DROP}")
        
        # Drop features
        df.drop(columns=FEATURES_TO_DROP, inplace=True, errors='ignore')
        
        # Log final state
        final_cols = df.columns.tolist()
        actually_dropped = set(initial_cols) - set(final_cols)
        logger.info(f"   • Features after selection: {len(final_cols)}")
        logger.info(f"   • Actually dropped: {list(actually_dropped)}")
        
        # Verify expected features remain
        missing_expected = set(FINAL_FEATURES) - set(final_cols)
        if missing_expected:
            logger.error(f"   ❌ Expected features missing: {missing_expected}")
            raise ValueError(f"Expected features missing after selection: {missing_expected}")
        
        logger.info("✅ Feature selection completed successfully")
        logger.info(f"   • Final feature count: {len(final_cols)} (14 features + 1 target)")
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Error during feature selection: {e}")
        raise RuntimeError(f"Feature selection failed: {e}") from e


def feature_scaling(
    df: pd.DataFrame, 
    method: str = 'minmax',
    save: bool = False
) -> pd.DataFrame:
    """
    Scale numerical features to a standard range.
    
    Uses MinMaxScaler (default) or StandardScaler.
    Only scales continuous numerical features (tenure, MonthlyCharges, TotalCharges).
    Does NOT scale the target variable (Churn).
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): Scaling method ('minmax' or 'standard')
        save (bool): If True, save scaler to disk
    
    Returns:
        pd.DataFrame: Dataframe with scaled features
    
    Raises:
        ValueError: If invalid scaling method
        RuntimeError: If scaling fails
    """
    logger.info(f"📏 Starting feature scaling (method: {method})...")
    
    try:
        # Validate method
        if method not in ['minmax', 'standard']:
            raise ValueError(f"Invalid method '{method}'. Must be 'minmax' or 'standard'")
        
        # Create scaler
        if method == 'standard':
            scaler = StandardScaler()
            logger.info("   • Using StandardScaler (mean=0, std=1)")
        else:
            scaler = MinMaxScaler()
            logger.info("   • Using MinMaxScaler (range=[0, 1])")
        
        # Copy dataframe
        df = df.copy(deep=True)
        
        # Identify columns to scale (must exist in dataframe)
        cols_to_scale = [col for col in FEATURES_TO_SCALE if col in df.columns]
        
        if not cols_to_scale:
            logger.warning("⚠️  No numerical columns found to scale")
            return df
        
        logger.info(f"   • Columns to scale: {cols_to_scale}")
        
        # Exclude target if present
        if TARGET_COLUMN in cols_to_scale:
            cols_to_scale.remove(TARGET_COLUMN)
            logger.info(f"   • Excluded target '{TARGET_COLUMN}' from scaling")
        
        # Log before scaling
        logger.info("   • Before scaling:")
        for col in cols_to_scale:
            logger.info(f"     - {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
        
        # Fit and transform
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
        # Log after scaling
        logger.info("   • After scaling:")
        for col in cols_to_scale:
            logger.info(f"     - {col}: [{df[col].min():.2f}, {df[col].max():.2f}]")
        
        # Save scaler if requested
        if save:
            logger.info("💾 Saving scaler...")
            PREPROCESSORS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = PREPROCESSORS_DIR / SCALER_FILE
            joblib.dump(scaler, save_path)
            logger.info(f"✅ Scaler saved to: {save_path}")
        
        logger.info("✅ Feature scaling completed successfully")
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Error during feature scaling: {e}")
        raise RuntimeError(f"Feature scaling failed: {e}") from e


def feature_engineering_pipeline(
    processed_data_path: Path,
    save: bool = False,
    scaling_method: str = 'minmax'
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Steps:
    1. Load preprocessed data
    2. Create new features (placeholder)
    3. Transform features (placeholder)
    4. Select relevant features
    5. Scale numerical features
    6. Save engineered data
    
    Args:
        processed_data_path (Path): Path to preprocessed data
        save (bool): If True, save scaler and engineered data
        scaling_method (str): Scaling method ('minmax' or 'standard')
    
    Returns:
        pd.DataFrame: Engineered dataframe
    
    Raises:
        FileNotFoundError: If preprocessed data not found
        RuntimeError: If pipeline fails
    """
    logger.info("=" * 80)
    logger.info("🚀 Starting Feature Engineering Pipeline")
    logger.info("=" * 80)
    
    try:
        # 1. Load preprocessed data
        if not processed_data_path.exists():
            logger.error(f"❌ Preprocessed data not found at: {processed_data_path}")
            raise FileNotFoundError(f"Preprocessed data not found: {processed_data_path}")
        
        logger.info(f"📂 Loading preprocessed data from: {processed_data_path}")
        df = pd.read_csv(processed_data_path)
        logger.info(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # 2. Feature creation
        df = feature_creation(df)
        
        # 3. Feature transformation
        df = feature_transformation(df)
        
        # 4. Feature selection
        df = feature_selection(df)
        
        # 5. Feature scaling
        df = feature_scaling(df, method=scaling_method, save=save)
        
        # 6. Save engineered data if requested
        if save:
            logger.info("💾 Saving engineered features...")
            FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
            feature_path = FEATURE_DATA_DIR / FEATURE_FILE_NAME
            df.to_csv(feature_path, index=False)
            logger.info(f"✅ Engineered data saved to: {feature_path}")
        
        # 7. Summary
        logger.info("\n" + "=" * 80)
        logger.info("📊 FEATURE ENGINEERING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"   • Final shape: {df.shape}")
        logger.info(f"   • Features: {df.shape[1] - 1} (+ 1 target)")
        logger.info(f"   • Columns: {list(df.columns)}")
        logger.info(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info(f"   • Missing values: {df.isnull().sum().sum()}")
        logger.info("=" * 80)
        
        logger.info("✅ Feature engineering pipeline completed successfully")
        
        return df
    
    except Exception as e:
        logger.error(f"❌ Feature engineering pipeline failed: {e}")
        raise RuntimeError(f"Feature engineering pipeline failed: {e}") from e


def main():
    """Main function for testing feature engineering module"""
    try:
        # Path to preprocessed data
        processed_path = PROCESSED_DATA_DIR / PROCESSED_FILE_NAME
        
        # Run feature engineering pipeline
        df_features = feature_engineering_pipeline(
            processed_path,
            save=True,
            scaling_method='minmax'
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("📊 ENGINEERED FEATURES PREVIEW")
        print("=" * 80)
        print(df_features.head(10))
        
        print("\n" + "=" * 80)
        print("📈 FEATURE STATISTICS")
        print("=" * 80)
        print(df_features.describe())
        
        print("\n" + "=" * 80)
        print("🎯 TARGET DISTRIBUTION")
        print("=" * 80)
        print(df_features['Churn'].value_counts())
        print(f"\nChurn rate: {df_features['Churn'].mean()*100:.2f}%")
        
        print("\n" + "=" * 80)
        print("📋 DATA TYPES")
        print("=" * 80)
        print(df_features.dtypes)
        
    except Exception as e:
        logger.error(f"❌ Feature engineering test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()