# 🚀 Real-Time Customer Churn Prediction System

Production-ready ML system for predicting customer churn using Telco Customer Churn dataset from Kaggle.

## 📊 Project Status

- ✅ **Phase 1: Data Ingestion** - COMPLETED
- ✅ **Phase 2: Data Preprocessing** - COMPLETED
- ✅ **Phase 3: Feature Engineering** - COMPLETED
- ⏳ **Phase 4: Model Training** - PENDING
- ⏳ **Phase 5: Model Deployment** - PENDING

## 🗂️ Project Structure
```
customer-churn-prediction/
├── src/
│   ├── data_pipeline/
│   │   ├── data_ingestion.py          ✅ Kaggle data download & loading
│   │   ├── data_preprocessing.py      ✅ Cleaning, encoding, validation
│   │   └── feature_engineering.py     ✅ Selection, scaling, transformation
│   └── utils/
│       ├── logger.py                  ✅ Centralized logging
│       ├── config.py                  ✅ Configuration management
│       └── const.py                   ✅ Project constants
├── data/
│   ├── raw/                           ✅ Raw data from Kaggle (7043 rows)
│   ├── processed/                     ✅ Preprocessed data (encoded)
│   └── features/                      ✅ Engineered features (14 features)
├── artifacts/
│   └── preprocessors/                 ✅ Saved encoders & scaler
├── logs/                              ✅ Application logs
├── tests/                             ✅ Unit tests (26 tests passing)
└── main.py                            ✅ Main entry point
```

## 🔧 Features

### Phase 1: Data Ingestion ✅

- ✅ Download from Kaggle API
- ✅ Local caching (avoid re-download)
- ✅ Data validation (schema, missing values, duplicates)
- ✅ Comprehensive logging
- ✅ Error handling

### Phase 2: Data Preprocessing ✅

- ✅ Data cleaning (TotalCharges → numeric, 11 NaN → 0)
- ✅ Target encoding (Churn: Yes/No → 1/0)
- ✅ Categorical encoding (LabelEncoder for 16 features)
- ✅ Schema validation
- ✅ Encoder persistence (for production inference)

### Phase 3: Feature Engineering ✅

- ✅ Feature selection (dropped 7 non-predictive features)
- ✅ Feature scaling (MinMaxScaler on 3 numerical features)
- ✅ Feature creation (placeholder for future features)
- ✅ Feature transformation (placeholder for future transformations)
- ✅ Scaler persistence (for production inference)

## 🚀 Quick Start
```powershell
# Run full pipeline (Phase 1 + 2 + 3)
python main.py

# Test specific phase
python src/data_pipeline/data_ingestion.py
python src/data_pipeline/data_preprocessing.py
python src/data_pipeline/feature_engineering.py

# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_ingestion.py -v
pytest tests/test_preprocessing.py -v
pytest tests/test_feature_engineering.py -v
```

## 📊 Dataset Transformation

| Phase | Shape | Description |
|-------|-------|-------------|
| **Phase 1: Raw** | 7043 × 21 | Raw data from Kaggle |
| **Phase 2: Preprocessed** | 7043 × 21 | All features encoded |
| **Phase 3: Features** | 7043 × 14 | Selected & scaled features |

**Final Features (14):**
1. SeniorCitizen (binary)
2. Partner (encoded)
3. Dependents (encoded)
4. tenure (scaled [0,1])
5. OnlineSecurity (encoded)
6. OnlineBackup (encoded)
7. DeviceProtection (encoded)
8. TechSupport (encoded)
9. Contract (encoded)
10. PaperlessBilling (encoded)
11. PaymentMethod (encoded)
12. MonthlyCharges (scaled [0,1])
13. TotalCharges (scaled [0,1])
14. **Churn (target)** 🎯

**Features Dropped (7):**
- customerID (unique identifier)
- gender (low correlation)
- PhoneService (redundant)
- MultipleLines (correlated)
- InternetService (redundant)
- StreamingTV (low impact)
- StreamingMovies (low impact)

**Target Distribution:**
- No Churn (0): 5174 (73.5%)
- Churn (1): 1869 (26.5%)

## 📦 Saved Artifacts
```
artifacts/preprocessors/
├── label_encoders.pkl  # 16 LabelEncoders for categorical features
└── scaler.pkl          # MinMaxScaler for numerical features
```

These artifacts are **production-ready** and ensure consistent preprocessing for:
- Training data
- Validation data
- **Production inference** (new customers)

## 🧪 Testing
```powershell
# Test all
pytest tests/ -v

# Test with coverage
pytest tests/ -v --cov=src

# Test specific module
pytest tests/test_feature_engineering.py -v
```

**Current test status:**
- ✅ Ingestion: 5/5 tests passing
- ✅ Preprocessing: 9/9 tests passing
- ✅ Feature Engineering: 12/12 tests passing
- ✅ **Total: 26/26 tests passing**

## 📈 Next Steps

**Phase 4: Model Training**
- Random Forest Classifier
- XGBoost Classifier
- Neural Network (PyTorch)
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- SMOTE for imbalance handling
- MLflow tracking
- Model comparison & selection

**Phase 5: Model Deployment**
- MLflow Model Registry
- API deployment (FastAPI)
- Docker containerization
- CI/CD pipeline
- Monitoring & retraining

## 👨‍💻 Author

Real-Time Customer Churn Prediction System - 2024
```

---

## 🎉 FÉLICITATIONS ! Phase 3 Terminée !

### **✅ Récapitulatif Complet (Phases 1-2-3)**

| Phase | Status | Features | Tests |
|-------|--------|----------|-------|
| **Phase 1: Ingestion** | ✅ | Download, Load, Validate | 5/5 ✅ |
| **Phase 2: Preprocessing** | ✅ | Clean, Encode, Validate | 9/9 ✅ |
| **Phase 3: Feature Engineering** | ✅ | Select, Scale, Transform | 12/12 ✅ |
| **TOTAL** | ✅ | **Data Pipeline Complete** | **26/26 ✅** |

---

## 📊 Ce qui a été accompli

### **Data Transformation Journey:**
```
Raw Data (Kaggle)
  ↓ Phase 1: Ingestion
7043 rows × 21 columns (raw)
  ↓ Phase 2: Preprocessing
7043 rows × 21 columns (encoded)
  ↓ Phase 3: Feature Engineering
7043 rows × 14 columns (selected & scaled)
  ↓
Ready for ML Training! 🚀
```

### **Artifacts Sauvegardés:**
```
✅ data/processed/telco_churn_processed.csv
✅ data/features/telco_churn_features.csv
✅ artifacts/preprocessors/label_encoders.pkl (16 encoders)
✅ artifacts/preprocessors/scaler.pkl (MinMaxScaler)