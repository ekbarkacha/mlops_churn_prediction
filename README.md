
# 🚀 Customer Churn Prediction API - Refactored Architecture

## Overview

A modern, modular FastAPI API for customer churn prediction featuring complete MLOps capabilities:

  - ✅ Canary deployment
  - ✅ Real-time monitoring (Prometheus + Evidently)
  - ✅ ML Explainability (SHAP)
  - ✅ JWT Authentication
  - ✅ Rate limiting
  - ✅ Layered Architecture

-----

## Architecture

### Folder Structure

```
src/app/
├── 📄 main.py                    # Entry point (200 lines)
├── 📄 config.py                  # Central configuration
│
├── 📁 core/                      # Core functionalities
│   ├── security.py              # Password hash, JWT, API keys
│   ├── deps.py                  # FastAPI Dependencies
│   └── middleware.py            # Logging, monitoring middleware
│
├── 📁 models/                    # Pydantic Schemas
│   ├── user.py                  # User schemas
│   ├── prediction.py            # Prediction schemas
│   ├── monitoring.py            # Monitoring schemas
│   └── common.py                # Common schemas
│
├── 📁 api/v1/                    # API Routes v1
│   ├── auth.py                  # Authentication endpoints
│   ├── users.py                 # User management endpoints
│   ├── predictions.py           # Prediction endpoints
│   ├── monitoring.py            # Monitoring endpoints
│   └── admin.py                 # Admin endpoints
│
├── 📁 services/                  # Business Logic Layer
│   ├── user_service.py          # User CRUD operations
│   ├── model_service.py         # ML model management
│   ├── prediction_service.py    # Prediction logic
│   └── monitoring_service.py    # Monitoring & metrics
│
├── 📁 ml/                        # ML Components
│   ├── model_wrapper.py         # Universal MLflow wrapper
│   └── preprocessing.py         # Data preprocessing
│
└── 📁 utils/                     # Utility functions
    └── csv_utils.py
```

### Layered Architecture

```
┌─────────────────────────────────────────┐
│          API Routes (api/v1/)           │  ← HTTP Endpoints
├─────────────────────────────────────────┤
│       Business Logic (services/)        │  ← Business Logic
├─────────────────────────────────────────┤
│     ML Operations (ml/)                 │  ← ML Workflows
├─────────────────────────────────────────┤
│     Data Layer (utils/, config)         │  ← Data Access
└─────────────────────────────────────────┘
```

-----

## ⚡ Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your values
```

### Launch the API

```bash
# Option 1: Direct Python
python src/app/main.py

# Option 2: Uvicorn
uvicorn src.app.main:app --reload --host 0.0.0.0 --port 8000

# Option 3: Docker
docker-compose up -d
```

### Access Documentation

  - **Swagger UI:** `http://localhost:8000/docs`
  - **ReDoc:** `http://localhost:8000/redoc`
  - **Health Check:** `http://localhost:8000/health`
  - **Metrics:** `http://localhost:8000/metrics`

-----

## Endpoints API

### 🔐 Authentication

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/register` | Register new user | API Key |
| POST | `/token` | Login and get JWT | API Key |
| GET | `/verify-session` | Verify JWT validity | API Key + JWT |

### 👥 User Management (Admin only)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/users` | List all users | API Key + JWT (Admin) |
| POST | `/users/approve` | Approve/disapprove user | API Key + JWT (Admin) |

### 🤖 Predictions

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| POST | `/predict` | Make predictions | API Key + JWT |
| POST | `/predict/explainability` | Get SHAP values | API Key + JWT (Admin) |

### 📊 Monitoring (Admin only)

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/monitoring/data_monitoring` | Drift detection report | API Key + JWT (Admin) |
| POST | `/monitoring/update_metrics` | Update Prometheus metrics | API Key |
| GET | `/monitoring/all_version_metrics` | Get all model versions | API Key + JWT (Admin) |

### ⚙️ Admin Operations

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/admin/feedback_data` | Get inference logs | API Key + JWT (Admin) |
| POST | `/admin/feedback` | Upload feedback data | API Key + JWT (Admin) |
| POST | `/admin/upload_data` | Upload training data | API Key + JWT (Admin) |
| POST | `/admin/set_canary_percentage` | Set canary weight | Admin API Key |
| POST | `/admin/reload_models` | Reload models | Admin API Key |

-----

## Usage Examples

### 1\. Authentication

```python
import requests

# Register
response = requests.post(
    "http://localhost:8000/register",
    headers={"x-api-key": "your-api-key"},
    json={
        "username": "john_doe",
        "password": "secure_password",
        "role": "agent"
    }
)

# Login
response = requests.post(
    "http://localhost:8000/token",
    headers={"x-api-key": "your-api-key"},
    data={
        "username": "john_doe",
        "password": "secure_password"
    }
)
# The JWT token is in the "access_token" cookie
```

### 2\. Prediction (Agent)

```python
# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    headers={"x-api-key": "your-api-key"},
    cookies={"access_token": "your-jwt-token"},
    json={
        "customerID": "1234-ABCD",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        # ... other features
    }
)

print(response.json())
# {"role": "agent", "prediction": 1, "probability": 0.75}
```

### 3\. Batch Prediction with SHAP (Admin)

```python
import pandas as pd

# Prepare CSV
df = pd.DataFrame([...])  # Your data

# Upload CSV
files = {"file": ("customers.csv", df.to_csv(index=False))}
response = requests.post(
    "http://localhost:8000/predict",
    headers={"x-api-key": "your-api-key"},
    cookies={"access_token": "admin-jwt-token"},
    files=files
)

result = response.json()
# result["results"] = predictions + probabilities
# result["shap_values"] = SHAP explanations
```

### 4\. Drift Monitoring

```python
response = requests.get(
    "http://localhost:8000/monitoring/data_monitoring?format=html&window=100",
    headers={"x-api-key": "your-api-key"},
    cookies={"access_token": "admin-jwt-token"}
)

# Save the HTML report
with open("drift_report.html", "w") as f:
    f.write(response.text)
```

-----

## Services

### UserService

Manages authentication and users:

  - Loading/saving users (JSON)
  - Account creation
  - Authentication
  - Admin approval

<!-- end list -->

```python
from src.app.services.user_service import UserService

user_service = UserService()
users = user_service.get_all_users()
user_service.approve_user("john_doe", approve=True)
```

### ModelService

Manages ML models:

  - Loading from MLflow
  - Canary deployment
  - Local model caching
  - Production/Staging selection

<!-- end list -->

```python
from src.app.services.model_service import ModelService

model_service = ModelService()
await model_service.load_models()
model, stage, version = model_service.get_model_for_inference()
model_service.set_canary_weight(0.1)  # 10% traffic to Staging
```

### PredictionService

Prediction logic:

  - Input validation
  - Predictions
  - Inference logging
  - Prometheus metrics

<!-- end list -->

```python
from src.app.services.prediction_service import PredictionService

pred_service = PredictionService()
preds, probs, shap = pred_service.make_predictions(model, df, include_shap=True)
pred_service.log_predictions_batch(raw_inputs, preds, probs, stage, version, latency)
```

### MonitoringService

Monitoring and metrics:

  - Reading/writing inference logs
  - Merging feedback
  - Updating Prometheus metrics

<!-- end list -->

```python
from src.app.services.monitoring_service import MonitoringService

mon_service = MonitoringService()
feedback_df = mon_service.get_feedback_data(limit=50)
mon_service.update_drift_metrics("churn_model", drift_share=0.15, drift_count=3)
```

-----

## ML Components

### InferencePreprocessor

Centralized preprocessing for inference:

```python
from src.app.ml.preprocessing import InferencePreprocessor

preprocessor = InferencePreprocessor(version_dir="models/v3")
df_clean = preprocessor.data_cleaning(df)
df_encoded = preprocessor.encode_categorical(df_clean)
df_processed = preprocessor.preprocess(df)  # All-in-one
```

### UniversalMLflowWrapper

Universal wrapper for sklearn, XGBoost, PyTorch:

```python
from src.app.ml.model_wrapper import UniversalMLflowWrapper

wrapper = UniversalMLflowWrapper(
    model=sklearn_model,
    version_dir="models/v3",
    background_data=train_sample
)

preds, probs = wrapper.predict(df, return_proba=True, both=True)
shap_values = wrapper.explain_json(df)
```

-----

## Configuration

### Environment Variables (.env)

```bash
# API Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
API_KEY=your-api-key
ADMIN_API_KEY=your-admin-api-key
ACCESS_TOKEN_EXPIRE_DAYS=30

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Paths (optional, defaults exist)
USERS_DATA_DIR=data/users
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
INFERENCE_DATA_DIR=data/inference
MODEL_DIR=models
```

-----

## Tests

```bash
# Install pytest
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/app --cov-report=html
```

## Development

### Adding a New Endpoint

1.  **Create the schema** in `models/`
2.  **Add the logic** in `services/`
3.  **Create the route** in `api/v1/`
4.  **Register the router** in `main.py`

Example:

```python
# 1. models/custom.py
class CustomRequest(BaseModel):
    data: str

# 2. services/custom_service.py
class CustomService:
    def process(self, data: str):
        return {"result": data.upper()}

# 3. api/v1/custom.py
router = APIRouter(prefix="/custom", tags=["Custom"])

@router.post("")
async def custom_endpoint(request: CustomRequest):
    service = CustomService()
    return service.process(request.data)

# 4. main.py
from src.app.api.v1 import custom
app.include_router(custom.router)
```

## Troubleshooting

### Models Are Not Loading

  - Check `MLFLOW_TRACKING_URI` in `.env`
  - Verify that MLflow is accessible
  - Check logs: `tail -f logs/app.log`

### Authentication Errors

  - Check `API_KEY` and `SECRET_KEY` in `.env`
  - Verify that the user is approved
  - Verify that the JWT has not expired

### Preprocessing Errors

  - Verify that columns match `EXPECTED_COLUMNS`
  - Verify that preprocessors exist in `models/vX/preprocessors/`

## Performance

  - **Rate Limiting:** 5/min for auth, 10/min for predictions
  - **Background Tasks:** Asynchronous logging
  - **Caching:** Models and preprocessors cached locally
  - **Lazy Loading:** SHAP explainer initialized on demand

## Security

  - ✅ Passwords hashed with `pwdlib`
  - ✅ JWT with expiration
  - ✅ API keys for all routes
  - ✅ HttpOnly cookies
  - ✅ CORS configured
  - ✅ Rate limiting
  - ✅ Input validation with Pydantic

## Monitoring

### Prometheus Metrics

  - `prediction_count`: Number of predictions
  - `prediction_latency`: Prediction latency
  - `prediction_errors`: Prediction errors
  - `drift_share`: Share of drifting features
  - `model_accuracy`: Model accuracy
  - ... and more

### Structured Logs

```json
{
  "id": "uuid",
  "method": "POST",
  "path": "/predict",
  "status": 200,
  "duration": 0.123
}
```

## License

Proprietary - AIMS-AMMI Student Project

## Support

For any questions:

  - 📖 Check `/docs` for interactive documentation
  - 📝 Read `REFACTORING_GUIDE.md` for migration details
  - 🐛 Open an issue on GitHub