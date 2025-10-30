# training.py
# This script handles loading prepared data, hyperparameter optimization with Optuna,
# model training, threshold tuning, evaluation, calibration, saving the model,
# and generating reports/visualizations.
# Designed for VSCode/local environment.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history
import pickle
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Directory where artifacts are stored
ARTIFACTS_DIR = "artifacts"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_prepared_data():
    """Load the prepared datasets and artifacts."""
    X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'X_train_scaled.csv'))
    X_val = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'X_val_scaled.csv'))
    X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'X_test_scaled.csv'))
    y_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'y_train.csv')).values.ravel()
    y_val = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'y_val.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'y_test.csv')).values.ravel()

    with open(os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl'), 'rb') as f:
        preprocessor = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)

    # Load resampled if exists, else handle here
    try:
        X_train_resampled = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'X_train_resampled.csv'))
        y_train_resampled = pd.read_csv(os.path.join(ARTIFACTS_DIR, 'y_train_resampled.csv')).values.ravel()
    except FileNotFoundError:
        X_train_resampled, y_train_resampled = None, None

    print("Prepared data loaded.")
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names, X_train_resampled, y_train_resampled

def handle_imbalance(X_train, y_train, X_val, y_val):
    """Handle class imbalance and select best strategy using validation set."""
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.combine import SMOTETomek
    from sklearn.metrics import roc_auc_score

    resampling_strategies = {
        'SMOTE': SMOTE(random_state=42, k_neighbors=5),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42, kind='borderline-1'),
        'ADASYN': ADASYN(random_state=42),
        'SMOTETomek': SMOTETomek(random_state=42),
    }

    best_strategy = None
    best_score = 0
    best_X_resampled = None
    best_y_resampled = None

    for name, sampler in resampling_strategies.items():
        try:
            X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
            quick_model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                        random_state=42, eval_metric='logloss', use_label_encoder=False)
            quick_model.fit(X_resampled, y_resampled)
            y_val_proba = quick_model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_val_proba)
            print(f"{name}: ROC-AUC={score:.4f}")
            if score > best_score:
                best_score = score
                best_strategy = name
                best_X_resampled = X_resampled
                best_y_resampled = y_resampled
        except Exception as e:
            print(f"Error in {name}: {e}")

    print(f"Best resampling strategy: {best_strategy}")
    return best_X_resampled, best_y_resampled, best_strategy

def optuna_optimization(X_train_resampled, y_train_resampled, X_val, y_val):
    """Optuna hyperparameter optimization focusing on F1-score."""
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 2.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'tree_method': 'hist'
        }
        model = XGBClassifier(**params)
        model.fit(X_train_resampled, y_train_resampled)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.3, 0.7, 0.05)
        best_f1 = 0
        for thresh in thresholds:
            y_val_pred = (y_val_proba >= thresh).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            if f1 > best_f1:
                best_f1 = f1
        return best_f1

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"Best F1-Score: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    return study

def train_optimized_model(study, X_train_resampled, y_train_resampled):
    """Train the optimized XGBoost model."""
    best_params = study.best_params
    model = XGBClassifier(**best_params)
    model.fit(X_train_resampled, y_train_resampled)
    print("Optimized model trained.")
    return model

def threshold_tuning(model, X_val, y_val):
    """Tune the decision threshold on validation set."""
    y_val_proba = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.2, 0.8, 0.01)
    metrics_by_threshold = []
    for threshold in thresholds:
        y_val_pred = (y_val_proba >= threshold).astype(int)
        metrics_by_threshold.append({
            'threshold': threshold,
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision': precision_score(y_val, y_val_pred),
            'recall': recall_score(y_val, y_val_pred),
            'f1': f1_score(y_val, y_val_pred)
        })
    threshold_df = pd.DataFrame(metrics_by_threshold)
    optimal_f1_idx = threshold_df['f1'].idxmax()
    optimal_f1_threshold = threshold_df.loc[optimal_f1_idx, 'threshold']

    high_f1_df = threshold_df[threshold_df['f1'] >= 0.75]
    if len(high_f1_df) > 0:
        optimal_acc_idx = high_f1_df['accuracy'].idxmax()
        optimal_balanced_threshold = high_f1_df.loc[optimal_acc_idx, 'threshold']
    else:
        optimal_balanced_threshold = optimal_f1_threshold

    best_threshold = optimal_balanced_threshold
    print(f"Selected threshold: {best_threshold:.3f}")
    return best_threshold, threshold_df

def evaluate_model(model, X_test, y_test, threshold):
    """Evaluate the model on test set."""
    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_proba),
        'avg_precision': average_precision_score(y_test, y_test_proba)
    }

    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nFinal Results:")
    for key, value in metrics.items():
        print(f"  {key.capitalize()}: {value:.4f}")

    print(classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn'], digits=4))

    return metrics, cm, y_test_proba

def calibrate_model(model, X_train, y_train, X_test, y_test, threshold):
    """Calibrate the model and compare."""
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated_model.fit(X_train, y_train)

    y_test_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    y_test_pred_cal = (y_test_proba_cal >= threshold).astype(int)

    cal_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred_cal),
        'precision': precision_score(y_test, y_test_pred_cal),
        'recall': recall_score(y_test, y_test_pred_cal),
        'f1': f1_score(y_test, y_test_pred_cal),
        'roc_auc': roc_auc_score(y_test, y_test_proba_cal)
    }

    return calibrated_model, cal_metrics, y_test_proba_cal

def create_visualizations(model, X_train, X_test, y_test, y_test_proba, threshold_df, best_threshold, study, metrics, cm):
    """Create and save visualizations."""
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

    # ROC Curve
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    ax1.plot(fpr, tpr, 'b-', label=f'AUC={metrics["roc_auc"]:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Precision-Recall Curve
    ax2 = fig.add_subplot(gs[0, 1])
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_test_proba)
    ax2.plot(recall_curve, precision_curve, 'r-', label=f'AP={metrics["avg_precision"]:.3f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    ax3.set_title('Confusion Matrix')
    ax3.set_ylabel('True Label')
    ax3.set_xlabel('Predicted Label')

    # Threshold Analysis
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(threshold_df['threshold'], threshold_df['accuracy'], 'b-', label='Accuracy')
    ax4.plot(threshold_df['threshold'], threshold_df['precision'], 'g-', label='Precision')
    ax4.plot(threshold_df['threshold'], threshold_df['recall'], 'r-', label='Recall')
    ax4.plot(threshold_df['threshold'], threshold_df['f1'], 'm-', label='F1-Score')
    ax4.axvline(x=best_threshold, color='black', linestyle='--', label=f'Optimal ({best_threshold:.3f})')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title('Threshold Optimization')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_ylim([0.4, 1.0])

    # Feature Importance
    ax5 = fig.add_subplot(gs[2, :])
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    ax5.barh(range(len(indices)), importances[indices])
    ax5.set_yticks(range(len(indices)))
    ax5.set_yticklabels([X_train.columns[i] for i in indices])
    ax5.set_xlabel('Importance')
    ax5.set_title('Top 20 Features')
    ax5.grid(alpha=0.3)

    # Probability Distribution
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.hist(y_test_proba[y_test == 0], bins=50, alpha=0.7, label='No Churn', color='blue')
    ax6.hist(y_test_proba[y_test == 1], bins=50, alpha=0.7, label='Churn', color='red')
    ax6.axvline(x=best_threshold, color='green', linestyle='--')
    ax6.set_xlabel('Probability')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Probability Distribution')
    ax6.legend()
    ax6.grid(alpha=0.3)

    # Performance vs Targets
    ax7 = fig.add_subplot(gs[3, 1])
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    metrics_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['roc_auc']]
    targets_values = [0.80] * 5
    x = np.arange(len(metrics_names))
    width = 0.35
    ax7.bar(x - width/2, metrics_values, width, label='Achieved')
    ax7.bar(x + width/2, targets_values, width, label='Target', alpha=0.6)
    ax7.set_ylabel('Score')
    ax7.set_title('Performance vs Targets')
    ax7.set_xticks(x)
    ax7.set_xticklabels(metrics_names, rotation=45)
    ax7.legend()
    ax7.grid(alpha=0.3)
    ax7.set_ylim([0, 1.0])

    # Optuna History
    ax8 = fig.add_subplot(gs[3, 2])
    trials_df = study.trials_dataframe()
    ax8.plot(trials_df['number'], trials_df['value'], 'o-')
    ax8.axhline(y=study.best_value, color='red', linestyle='--')
    ax8.set_xlabel('Trial')
    ax8.set_ylabel('F1-Score')
    ax8.set_title('Optuna Progress')
    ax8.grid(alpha=0.3)

    plt.savefig(os.path.join(MODELS_DIR, 'advanced_model_evaluation.png'), dpi=300, bbox_inches='tight')
    print("Visualizations saved.")

def save_model_and_artifacts(final_model, best_threshold, feature_names, metrics, study, best_strategy, X_train, calibrated=False):
    """Save the final model and metadata."""
    with open(os.path.join(MODELS_DIR, 'champion_model_advanced.pkl'), 'wb') as f:
        pickle.dump(final_model, f)
    with open(os.path.join(MODELS_DIR, 'optimal_threshold.pkl'), 'wb') as f:
        pickle.dump(best_threshold, f)
    with open(os.path.join(MODELS_DIR, 'feature_names_advanced.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)

    metadata = {
        'model_name': 'XGBoost Advanced',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'optimal_threshold': best_threshold,
        'test_metrics': metrics,
        'features': feature_names,
        'n_features': len(feature_names),
        'resampling_strategy': best_strategy,
        'optuna_trials': len(study.trials),
        'best_optuna_f1': study.best_value,
        'hyperparameters': study.best_params,
        'calibrated': calibrated
    }
    with open(os.path.join(MODELS_DIR, 'model_metadata_advanced.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    print("Model and artifacts saved.")

# Prediction function (define advanced_feature_engineering if not imported)
def advanced_feature_engineering(X):
    # Copy the function from data_ingestion.py here
    X_new = X.copy()
    cols = X_new.columns.tolist()
    CLIP_THRESHOLD = -0.9999

    if 'MonthlyCharges' in cols and 'tenure' in cols:
        X_new['estimated_ltv'] = X_new['MonthlyCharges'] * X_new['tenure']
        X_new['monthly_charge_growth'] = X_new['MonthlyCharges'] / (X_new['tenure'] + 1)
        if 'TotalCharges' in cols:
            expected_total = X_new['MonthlyCharges'] * X_new['tenure']
            X_new['charge_inefficiency'] = (X_new['TotalCharges'] - expected_total) / (expected_total + 1)

    if 'tenure' in cols:
        X_new['tenure_log'] = np.log1p(np.maximum(CLIP_THRESHOLD, X_new['tenure']))
        X_new['tenure_squared'] = X_new['tenure'] ** 2
        X_new['is_very_new'] = (X_new['tenure'] <= 6).astype(int)
        X_new['is_new'] = ((X_new['tenure'] > 6) & (X_new['tenure'] <= 12)).astype(int)
        X_new['is_established'] = ((X_new['tenure'] > 12) & (X_new['tenure'] <= 36)).astype(int)
        X_new['is_loyal'] = (X_new['tenure'] > 36).astype(int)

    if 'MonthlyCharges' in cols:
        X_new['MonthlyCharges_log'] = np.log1p(np.maximum(CLIP_THRESHOLD, X_new['MonthlyCharges']))
        X_new['is_high_spender'] = (X_new['MonthlyCharges'] > X_new['MonthlyCharges'].quantile(0.75)).astype(int)
        X_new['is_low_spender'] = (X_new['MonthlyCharges'] < X_new['MonthlyCharges'].quantile(0.25)).astype(int)

    service_cols = [c for c in cols if '_Yes' in c or 'PhoneService' in c or 'InternetService' in c]
    if service_cols:
        X_new['active_services_count'] = X_new[service_cols].sum(axis=1)
        X_new['has_no_services'] = (X_new['active_services_count'] == 0).astype(int)
        X_new['has_many_services'] = (X_new['active_services_count'] >= 3).astype(int)

    if 'MonthlyCharges' in cols and 'active_services_count' in X_new.columns:
        X_new['price_per_service'] = X_new['MonthlyCharges'] / (X_new['active_services_count'] + 1)
        median_pps = X_new['price_per_service'].median()
        X_new['is_overpriced'] = (X_new['price_per_service'] > median_pps * 1.5).astype(int)

    if 'SeniorCitizen' in cols:
        X_new['SeniorCitizen_squared'] = X_new['SeniorCitizen'] ** 2

    partner_col = [c for c in cols if 'Partner_Yes' in c]
    dependents_col = [c for c in cols if 'Dependents_Yes' in c]
    if partner_col and dependents_col:
        X_new['family_score'] = X_new[partner_col[0]] + X_new[dependents_col[0]]
        if 'SeniorCitizen' in cols:
            X_new['senior_alone'] = ((X_new['SeniorCitizen'] == 1) & (X_new['family_score'] == 0)).astype(int)

    contract_cols = [c for c in cols if 'Contract_' in c]
    payment_cols = [c for c in cols if 'PaymentMethod_' in c]
    month_to_month = [c for c in contract_cols if 'Month-to-month' in c]
    if month_to_month:
        X_new['risky_contract'] = X_new[month_to_month[0]]
    electronic_check = [c for c in payment_cols if 'Electronic' in c]
    if electronic_check and 'risky_contract' in X_new.columns:
        X_new['high_risk_profile'] = ((X_new['risky_contract'] == 1) & (X_new[electronic_check[0]] == 1)).astype(int)

    return X_new

def predict_churn(customer_data, model, preprocessor, feature_names, threshold):
    """Prediction function for production."""
    customer_enhanced = advanced_feature_engineering(customer_data)
    for feature in feature_names:
        if feature not in customer_enhanced.columns:
            customer_enhanced[feature] = 0
    customer_enhanced = customer_enhanced[feature_names]
    # Assuming preprocessor is for scaling new data, but since saved data is scaled, for new raw data, apply preprocessing
    # Note: For production, need full pipeline; here assume customer_data is raw, but adjust as needed
    # customer_processed = preprocessor.transform(customer_enhanced)  # If needed
    proba = model.predict_proba(customer_enhanced)[:, 1][0]
    prediction = 1 if proba >= threshold else 0
    return {
        'prediction': 'Churn' if prediction == 1 else 'No Churn',
        'probability': float(proba),
        'confidence': abs(proba - 0.5) * 2,
        'risk_level': 'High' if proba > 0.7 else 'Medium' if proba > 0.4 else 'Low'
    }

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, feature_names, X_train_resampled, y_train_resampled = load_prepared_data()

    if X_train_resampled is None or y_train_resampled is None:
        X_train_resampled, y_train_resampled, best_strategy = handle_imbalance(X_train, y_train, X_val, y_val)
    else:
        best_strategy = "Loaded from artifacts"  # Placeholder

    study = optuna_optimization(X_train_resampled, y_train_resampled, X_val, y_val)
    model = train_optimized_model(study, X_train_resampled, y_train_resampled)
    best_threshold, threshold_df = threshold_tuning(model, X_val, y_val)
    metrics, cm, y_test_proba = evaluate_model(model, X_test, y_test, best_threshold)

    calibrated_model, cal_metrics, y_test_proba_cal = calibrate_model(model, X_train, y_train, X_test, y_test, best_threshold)

    if cal_metrics['f1'] > metrics['f1']:
        final_model = calibrated_model
        final_metrics = cal_metrics
        y_test_proba = y_test_proba_cal
        calibrated = True
        print("Selected calibrated model.")
    else:
        final_model = model
        final_metrics = metrics
        calibrated = False
        print("Selected original model.")

    create_visualizations(model, X_train, X_test, y_test, y_test_proba, threshold_df, best_threshold, study, final_metrics, cm)
    save_model_and_artifacts(final_model, best_threshold, feature_names, final_metrics, study, best_strategy, X_train, calibrated)

    # Save prediction function
    with open(os.path.join(MODELS_DIR, 'predict_function.pkl'), 'wb') as f:
        pickle.dump(predict_churn, f)

    # Test prediction
    sample_customer = X_test.iloc[[0]]  # Already enhanced and scaled
    result = predict_churn(sample_customer, final_model, preprocessor, feature_names, best_threshold)
    print("Prediction test:", result)