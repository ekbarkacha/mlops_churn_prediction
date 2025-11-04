import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import joblib
from sklearn.feature_selection import f_classif, chi2
from scipy import stats
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.const import (
    PROCESSED_DATA_DIR, MODEL_DIR,
    processed_file_name, feature_file_name, scaler_file_name
)

logger = get_logger(__name__)

# ===============================================================
#  Step 1: Load Processed Data                                  #
# ===============================================================
def load_processed_data() -> pd.DataFrame:
    """
    Loads the processed dataset (already encoded and cleaned).
    """
    path = os.path.join(PROCESSED_DATA_DIR, processed_file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed data not found at {path}")
    df = pd.read_csv(path)
    logger.info(f"Processed data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ===============================================================
#  Step 2: Statistical Feature Selection                        #
# ===============================================================
def statistical_feature_selection(df: pd.DataFrame, target_col="Churn", alpha=0.05):
    """
    Performs automatic feature selection using statistical significance tests.
    - If target is categorical (classification):
        * Numerical features: ANOVA F-test
        * Discrete features: Chi2 test
    - If target is numeric (regression):
        * Numerical features: Pearson correlation
        * Discrete features: ANOVA
    Keeps only features with p-value < alpha.
    """
    logger.info("Starting statistical feature selection...")

    try:
        y = df[target_col]
        X = df.drop(columns=[target_col])

        # Distinguish continuous vs discrete features
        num_cols = [col for col in X.columns if X[col].nunique() > 10]
        disc_cols = [col for col in X.columns if X[col].nunique() <= 10]

        selected_features = []

        # Classification case
        if len(np.unique(y)) <= 10:
            logger.info("Detected classification target → using ANOVA and Chi2")

            if len(num_cols) > 0:
                f_vals, p_vals = f_classif(X[num_cols], y)
                selected_features += [col for col, p in zip(num_cols, p_vals) if p < alpha]

            if len(disc_cols) > 0:
                chi_vals, p_vals = chi2(X[disc_cols], y)
                selected_features += [col for col, p in zip(disc_cols, p_vals) if p < alpha]

        # Regression case
        else:
            logger.info("Detected numeric target → using Pearson and ANOVA")

            for col in num_cols:
                corr, pval = stats.pearsonr(X[col], y)
                if pval < alpha:
                    selected_features.append(col)

            for col in disc_cols:
                groups = [y[X[col] == val] for val in X[col].unique()]
                if len(groups) > 1:
                    f, p = stats.f_oneway(*groups)
                    if p < alpha:
                        selected_features.append(col)

        selected_features = list(set(selected_features))
        logger.info(f"Selected {len(selected_features)} significant features.")
        return df[selected_features + [target_col]]

    except Exception as e:
        logger.error(f"Error during statistical feature selection: {e}")
        raise RuntimeError(f"Feature selection failed: {e}")


# ===============================================================
#  Step 3: Feature Scaling                                      #
# ===============================================================
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes numeric features for model training.
    """
    logger.info("Scaling features...")

    try:
        scaler = StandardScaler()
        num_cols = [col for col in df.select_dtypes(include=np.number).columns if col != "Churn"]
        df[num_cols] = scaler.fit_transform(df[num_cols])

        scaler_path = os.path.join(MODEL_DIR, scaler_file_name)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")

        return df

    except Exception as e:
        logger.error(f"Error during feature scaling: {e}")
        raise RuntimeError(f"Feature scaling failed: {e}")


# ===============================================================
#  Step 4: Full Feature Engineering Pipeline                    #
# ===============================================================
def feature_engineering_pipeline(save=True):
    """
    Full pipeline:
      - Load processed data
      - Perform statistical feature selection
      - Scale numeric features
      - Save reduced dataset
    """
    logger.info("Starting feature engineering pipeline...")

    df = load_processed_data()
    df = statistical_feature_selection(df, target_col="Churn")
    df = scale_features(df)

    if save:
        feature_path = os.path.join(PROCESSED_DATA_DIR, feature_file_name)
        os.makedirs(os.path.dirname(feature_path), exist_ok=True)
        df.to_csv(feature_path, index=False)
        logger.info(f"Feature-engineered data saved to {feature_path}")

    logger.info("Feature engineering completed successfully.")
    return df


# ===============================================================
#  Main Entry Point                                             #
# ===============================================================
def main():
    df = feature_engineering_pipeline(save=True)
    print(f"✅ Feature engineering done. Final shape: {df.shape}")

if __name__ == "__main__":
    main()
