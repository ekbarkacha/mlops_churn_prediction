#  Real-Time Customer Churn Prediction System (MLOps Project)

##  Overview
This project implements a **real-time customer churn prediction system** for a telecommunications company, following **end-to-end MLOps best practices**.  
The goal is to predict whether a customer will churn based on behavioral and demographic features, with **automated data pipelines, experiment tracking, CI/CD workflows, deployment, and monitoring**.

Customer churn is the phenomenon where customers discontinue their relationship with a company which poses a significant challenge to the telecommunications industry. High churn rates can lead to substantial revenue losses, reduced market share and hindered long-term growth. Therefore, accurately identifying customers who are likely to churn and intervening before they leave is critical for maintaining business stability and profitability.

This project focuses on building a Real-Time Customer Churn Prediction System powered by Machine Learning Operations (MLOps). The solution will continuously analyze customer data to predict churn probability and suggest what lead to it in real time. By integrating automated data pipelines, model retraining workflows, CI/CD orchestration, and inference APIs, the system ensures that predictions remain accurate, reliable and actionable.


The system integrates:
- Automated **data ingestion and preprocessing**
- **Model training and versioning** with MLflow and DVC
- **Continuous Integration & Deployment (CI/CD)** using GitHub Actions
- **FastAPI-based REST API** for model serving
- **Monitoring** with Prometheus and Airflow DAGs for drift and decay detection
- **Containerization** with Docker and deployment on **Azure Cloud**

---

## 🏗️ System Architecture / System Design Flow Diagram
The overall architecture consists of multiple interconnected components supporting the full machine learning lifecycle.

![System Design Flow](images/system_design_flow_diagram.png)

**Main Components:**
1. **Data Pipeline** – handles ingestion, cleaning, feature engineering, and data versioning.
2. **Model Development** – builds and evaluates multiple ML models with experiment tracking.
3. **CI/CD Pipeline** – automates testing, retraining, and deployment.
4. **Model Serving** – exposes predictions via a FastAPI REST endpoint.
5. **Monitoring & Feedback** – observes model performance, drift, and triggers retraining.

---

##  Technical Stack

| Category | Tools / Frameworks |
|-----------|--------------------|
| **Language** | Python 3.12 |
| **ML Libraries** | Scikit-learn, XGBoost, Joblib, etc |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC |
| **API Framework** | FastAPI |
| **Orchestration** | Apache Airflow |
| **Monitoring** | Prometheus + Grafana |
| **Containerization** | Docker & Docker Compose |
| **Cloud Platform** | Microsoft Azure |
| **CI/CD** | GitHub Actions |
| **Dependency Management** | requirements.txt |

---

##  Data Pipeline

The dataset used: [`WA_Fn-UseC_-Telco-Customer-Churn.csv`](data/raw/)

**Pipeline Steps:**
1. **Data Ingestion** (`src/data_pipeline/data_ingestion.py`)  
   - Reads data from CSV sources  
   - Validates schema and missing values  
   - Logs ingestion process  

2. **Preprocessing** (`src/data_pipeline/data_preprocessing.py`)  
   - Cleans missing and categorical data  
   - Encodes categorical variables (`label_encoder.joblib`)  
   - Normalizes numerical features (`scaler.joblib`)  

3. **Feature Engineering** (`src/data_pipeline/feature_engineering.py`)  
   - Creates customer usage and tenure-related features  
   - Saves processed data to `data/processed/`  

4. **Data Versioning**  
   - Managed through **DVC**, ensuring reproducibility of data and model artifacts  

---

##  Model Development

Model development occurs in the `src/modeling/` directory.

**Models Trained:**
- Logistic Regression  
- XGBoost  
- Neural Network (`nn_model.py`)

**Configuration File:**  
`model_config.yaml` defines hyperparameters, training parameters, and thresholds.

**Tracking:**
- **MLflow** is used to log metrics (Accuracy, F1-score, AUC) and model versions.
- Model artifacts and parameters are stored for reproducibility.

**Best Model Selection:**
- Models are compared based on **F1-score** and **AUC**.
- The final model is serialized and stored in `data/models/`.

---

##  CI/CD Pipeline

Implemented via **GitHub Actions**.

**Workflow Features:**
- Automatically triggered on code commits or pull requests.  
- Runs **unit tests** (e.g., `test/test_data_pipeline.py`) to validate data logic.  
- Performs **integration tests** to verify pipeline execution.  
- Triggers **model retraining** when new data is detected via DVC or Airflow.  
- Builds and deploys Docker containers for the API.

**Airflow DAGs:**
- `canary_deployment.py`: Gradually rolls out new model versions (5% → 25% → 100%).  
- `model_decay_data_drift.py`: Detects model degradation or drift and triggers retraining.

---

##  Deployment

Deployment is managed via **Docker** and **Azure Cloud**.

**Files:**
- `Dockerfile.fastapi`: Builds the FastAPI inference container  
- `Dockerfile.airflow`: Sets up Airflow for orchestration  
- `docker-compose-1.yml`, `docker-compose-2.yml`: Define service orchestration  

**FastAPI Application:**
- Located in `src/app/main.py`  
- Exposes endpoints for model prediction, retraining, and health checks  
- Includes authentication (`auth.py`) and configuration management (`config.py`)

Example Endpoint:
```bash
POST /predict
Content-Type: application/json
```
##  Monitoring & Observability

Monitoring is a key component of this MLOps pipeline, ensuring that the deployed model continues to perform as expected and that any degradation or data drift is detected early.

**Tools Used:**
- **Prometheus** → Collects key performance metrics such as latency, throughput, and error rates.  
- **Grafana** → Provides real-time dashboards for visualizing API and model metrics.  
- **Apache Airflow** → Automates periodic checks for data drift and model decay.

**Monitoring Scripts:**
- `check_data_drift.py` → Compares feature distributions between training and current data to detect drift.  
- `check_model_decay.py` → Evaluates model performance over time and flags degradation.  
- `check_canary.py` → Validates canary deployments before full rollout.

**Configuration File:**
- `prometheus.yml` → Defines metric scraping targets and alerting rules.

**Key Metrics Monitored:**
- API latency and uptime  
- Prediction accuracy and error rates  
- Feature drift statistics  
- Model performance trends over time

When a threshold breach or anomaly is detected, alerts are triggered to notify maintainers and optionally **initiate retraining pipelines**.

##  Security & Compliance

Security and compliance are integrated across all components of the MLOps system to protect customer data and ensure responsible model usage.

**Authentication & Authorization:**
- Implemented using **JWT-based tokens** via the `auth.py` module.
- **Role-Based Access Control (RBAC)** ensures only authorized users can access specific endpoints.

**Data Privacy:**
- Sensitive customer identifiers are **anonymized** before storage and model training.
- No raw personally identifiable information (PII) is exposed through the API or logs.

**Audit & Logging:**
- Every API request is logged through `utils/logger.py` for traceability.
- Access and activity logs are stored securely and periodically reviewed.

**Infrastructure Security:**
- All services are containerized via Docker, isolating runtime environments.
- Network-level access is restricted to trusted hosts within the Azure deployment.

**Compliance Alignment:**
- The system follows **GDPR-compliant** data handling practices.
- Access policies and monitoring align with standard **Azure security guidelines**.

---
## Repository Structure
```bash
mlops_churn_prediction/
│
├─ airflow/ 
|      └─ dag/             # Airflow DAGs for pipeline orchestration
│
├─ data/
│   └─ raw/                # Raw customer datasets from kaggle
│
├─ docs/                   # Documentation: product design
│
├─ images/                 # Images i.e system archecture
│
├─ scripts/                # Utility scripts for setting up airflow.
│
├─ src/                    # Main source code (data pipeline, model, API)
│
├─ test/                   # Unit and integration tests
│
├─ .github/workflows/      # CI/CD workflows (GitHub Actions)
│
├─ Dockerfile.airflow      # Dockerfile for Airflow orchestration
├─ Dockerfile.base         # Base Docker image for dependancies and system wich will be used in airflow and fastapi images
├─ Dockerfile.fastapi      # Dockerfile for FastAPI deployment
├─ Dockerfile.frontend     # Dockerfile for gradio app frontend
├─ docker-compose-1.yml    # Docker Compose setup for frontend, grafana and mlflow
├─ docker-compose-2.yml    # Docker Compose setup for fastapi,promethus.yml and airflow
├─ prometheus.yml          # Prometheus configuration
├─ requirements.txt        # Python dependencies
├─ README.md               # Project documentation
└─ .dvc/.dvcignore         # DVC files for data versioning

```

### Prerequisites

* Python 3.10+
* Docker & Docker Compose
* MLflow
* DVC
* Airflow
* Grafana
* Prometheus

### Installation

```bash
git clone https://github.com/ekbarkacha/mlops_churn_prediction.git
cd mlops_churn_prediction
pip install -r requirements.txt
```

### Run Locally

#### Docker-compose 1:
```bash
docker-compose -f docker-compose-1.yml up #Seting up mlflow, grafana and gradio app (frontend)
# Access Grafana API at http://localhost:3030
# Access Mlflow at http://localhost:5000
# Access Gradio app at http://localhost:7862
```

#### Docker-compose 2:
```bash
docker build -t ml-base:latest -f Dockerfile.base . # will be required in docker-compose-2.yml by airflow and fastapi
docker-compose -f docker-compose-2.yml up #Seting up airflow, prometheus and fastapi
# Access FastAPI API at http://localhost:8000
# Access Airflow at http://localhost:8080
# Access Prometheus at http://localhost:9090
```

## References

- [Kaggle Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)  
- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)  
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)  
- [Prometheus Monitoring](https://prometheus.io/docs/introduction/overview/)  
- [Grafana Dashboards](https://grafana.com/docs/grafana/latest/)  
- [DVC (Data Version Control)](https://dvc.org/doc)  
- [Azure Machine Learning Deployment Guide](https://learn.microsoft.com/en-us/azure/machine-learning/)  



