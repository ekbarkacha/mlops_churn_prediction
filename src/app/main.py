import os
import time, uuid, json
from fastapi import FastAPI, Depends, HTTPException,UploadFile,File,Request,BackgroundTasks
from fastapi.responses import JSONResponse,HTMLResponse,Response
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta, timezone
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from typing import Optional
from io import StringIO
from contextlib import asynccontextmanager
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
from mlflow.exceptions import MlflowException
from src.app.auth import (verify_api_key,get_password_hash,require_role,
                          authenticate_user,create_access_token,
                          set_middleware)
from src.app.schemas import User,CustomerInput
from src.app.utils import (load_users,save_users,get_current_user,
                           log_predictions_task,read_csv_safe,file_exist,
                           InferenceDatapreprocer)
from src.app.config import (APP_USERS,ACCESS_TOKEN_EXPIRE_DAYS,
                            MLFLOW_TRACKING_URI,MLFLOW_EXPERIMENT_NAME,
                            MODEL_DIR,EXPECTED_COLUMNS,INFERENCE_DATA_PATH,
                            PROCESSED_DATA_DIR,processed_file_name,RAW_DATA_PATH)

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from src.data_pipeline.data_preprocessing import validate_schema
from src.app.model_wrapper import UniversalMLflowWrapper
from src.utils.logger import get_logger

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, REGISTRY
from src.monitoring.metrics import (CURRENT_MODEL_VERSION,PREDICTION_COUNT,PREDICTION_LATENCY,
                                    PREDICTION_ERRORS,API_REQUEST_COUNT,API_LATENCY,API_ERRORS,
                                    DRIFT_SHARE_GAUGE, DRIFT_COUNT_GAUGE,PIPELINE_ERROR_COUNTER,
                                    MODEL_ACCURACY_GAUGE,MODEL_PRECISION_GAUGE,MODEL_RECALL_GAUGE,
                                    MODEL_ROC_AUC_GAUGE,MODEL_F1_GAUGE)

logger = get_logger("customer_churn_api")

# Setting mlflow tracking uri
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    - The latest MLflow model is automatically loaded and cached.
    - All preprocessing artifacts are downloaded.
    - A SHAP explainer is prepared for explainability.
    - Resources are properly released on shutdown.

    Runs once on application startup and cleanup on shutdown.
    """

    logger.info("Starting up FastAPI app and initializing ML model...")

    try:
        # Connect to MLflow Tracking Server
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        logger.info(f"Connected to MLflow at {MLFLOW_TRACKING_URI}")

        # Get all registered model versions
        versions = client.search_model_versions(f"name='{MLFLOW_EXPERIMENT_NAME}'")

    except MlflowException as e:
        logger.warning(f"Could not connect to MLflow: {e}")       
        versions = []

    if not versions:
        logger.warning(f"No model versions found for {MLFLOW_EXPERIMENT_NAME}. Skipping model load.")
        logger.info("App will start without a loaded model. Inference endpoints will return 503.")
        yield
        return

    # Sort model versions numerically and select the latest
    latest_version = sorted(versions, key=lambda v: int(v.version))[-1]
    model_uri = f"models:/{latest_version.name}/{latest_version.version}"
    version_folder = f"{MODEL_DIR}/v{latest_version.version}"

    # Get run name from the latest version
    run_info = client.get_run(latest_version.run_id)
    run_name = run_info.data.tags.get("mlflow.runName", "Unnamed Run")

    # Versioned model directory
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Download and cache model locally (if not cached)
    if not os.path.exists(version_folder):
        logger.info(f"Downloading model v{latest_version.version} from MLflow registry...")
       
       # Load model from MLflow
        model = mlflow.pyfunc.load_model(model_uri)

        # Save local copy for caching
        mlflow.pyfunc.save_model(
            path=version_folder,
            python_model=model._model_impl.python_model
        )

        # Download preprocessing artifacts (eg scaler, encoders)
        logger.info(f"Downloading preprocessor artifacts for run_id: {latest_version.run_id}")
        client.download_artifacts(run_id=latest_version.run_id, path="preprocessors", dst_path=version_folder)
        
        logger.info(f"Model version {latest_version.version} cached locally at {version_folder}.")
    else:
        logger.info(f"Loading cached model version {latest_version.version} from {version_folder}")
        model = mlflow.pyfunc.load_model(version_folder)
    
    # Load training data sample for SHAP explainability    
    logger.info("Preparing background SHAP data...")
    background = pd.read_csv(RAW_DATA_PATH)
    background = background[EXPECTED_COLUMNS[:-1]].sample(100, random_state=42)
    
    # Wrap model inside UniversalMLflowWrapper and Store in FastAPI app state
    app.state.model = UniversalMLflowWrapper(model._model_impl.python_model.model,
                                             version_dir=version_folder,
                                             background_data=background)
    
     # Store version directory for access by other endpoints
    app.state.version_folder = version_folder

    CURRENT_MODEL_VERSION.labels(model_name=MLFLOW_EXPERIMENT_NAME,model_type=run_name).set(latest_version.version)
    logger.info(f"Model {MLFLOW_EXPERIMENT_NAME} version {latest_version.version} loaded successfully.")
    yield

    logger.info("Shutting down app — releasing model resources...")
    if hasattr(app.state, "model"):
        del app.state.model
        logger.info("Model successfully unloaded.")

# App instantiated using lifespan 
app = FastAPI(lifespan=lifespan,title="Customer Churn Prediction API")

# Initialize metrics collector
instrumentator = Instrumentator().instrument(app)

# Expose /metrics endpoint for Prometheus scraping
instrumentator.expose(app)

# For rate limit
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def log_and_monitor_requests(request: Request, call_next):
    """
    Log and monitor all HTTP requests, except for ("/metrics","/verify-session","/update_metrics").
    """

    start = time.perf_counter()
    method = request.method
    endpoint = request.url.path

    # Skip Prometheus or internal endpoints
    if endpoint in ["/metrics","/verify-session","/update_metrics"]:
        return await call_next(request)

    try:
        # Generate or get request ID
        req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Process request
        response = await call_next(request)

        duration = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{duration:.3f}s"
        response.headers["X-Request-ID"] = req_id

        # Log request metadata
        log_data = {
            "id": req_id,
            "method": method,
            "path": endpoint,
            "status": response.status_code,
            "duration": round(duration, 3)
        }

        logger.info(json.dumps(log_data))

        # Update Prometheus metrics
        API_REQUEST_COUNT.labels(endpoint=endpoint, method=method).inc()
        API_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

        return response

    except Exception as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status_code=500).inc()
        logger.exception(f"{method} {endpoint} failed: {str(e)}")
        return Response("Internal server error", status_code=500)

# Allow requests from frontend
set_middleware(app)

# Auth Endpoints
@app.post("/register", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
def register(user: User,request: Request):
    """
    Registers a new user (requires API key).
    - Checks if username already exists.
    - Hashes password and saves username, password, and role to user database.
    """
    method = request.method
    endpoint = request.url.path
    try:
        users = load_users()
        if user.username in users:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Store role along with username and password
        hashed_password = get_password_hash(user.password)
        users[user.username] = {
            "username": user.username,
            "hashed_password": hashed_password,
            "role": user.role
        }
        save_users(users)
        logger.info(f"User {user.username} registered successfully.")
        return {"msg": "User created successfully"}
    except HTTPException as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status=e.status_code).inc()
        logger.error(f"Registration failed: {e.detail}")
        raise


@app.post("/token", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
def login(request: Request,form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticates user credentials and issues a JWT token.
    - Requires API key.
    - Validates username/password.
    - Generates JWT with embedded user role.
    - Sets token as an HttpOnly cookie for session persistence.
    """
    method = request.method
    endpoint = request.url.path
    try:
        user = authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(status_code=401, detail="Incorrect username or password")
        
        # Add role to JWT
        role = user.get("role", APP_USERS.get(1))
        
        token = create_access_token(
            data={"sub": user["username"], "role": role},
            expires_delta=timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
        )

        # Set token as HttpOnly cookie
        response = JSONResponse(content={"msg": "Login successful"})
        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            secure=False,  # True if using HTTPS
            samesite="Lax",
            max_age=ACCESS_TOKEN_EXPIRE_DAYS*24*60*60 # 30 days in seconds
        )
        logger.info(f"User {form_data.username} logged in successfully. Token issued.")
        return response
    except HTTPException as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status=e.status_code).inc()
        logger.warning(f"Login failed for {form_data.username}: {e.detail}")
        raise

@app.get("/verify-session", dependencies=[Depends(verify_api_key)])
def verify_session(user=Depends(get_current_user)):
    """
    Confirms whether the user's session (JWT token) is still valid.
    - Requires valid JWT and API key.
    - Returns the current username and role.
    """
    username = user["username"]
    role = user["role"]
    return {"username": username, "role": role}

# Model Endpoints
# 1. Prediction (Both Admin and Agent)
@limiter.limit("10/minute")
@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(
    request: Request,
     background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Generates churn predictions using the latest version of MLflow model.

    Admin:
    - Can upload CSV or send JSON data.
    - Validates schema, performs inference.
    - Returns predictions, probabilities, and SHAP explainability values.
    - Logs results as the production data

    Agent:
    - Must send single JSON input only.
    - Returns single prediction and probability.
    - Logs results as the production data
    """
    method = request.method
    endpoint = request.url.path
    start_time = time.perf_counter()
    if not hasattr(request.app.state, "model"):
        logger.warning("Prediction requested but no model is loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    try:
        logger.info(f"Prediction request received from user: {current_user['username']} ({current_user['role']})")
        model = request.app.state.model
        version = app.state.version_folder.split("/")[-1]

        # ADMIN ROLE
        if current_user["role"] == APP_USERS.get(2):
            df = None

            # CASE 1: File was uploaded
            if file is not None:
                logger.debug("CSV file uploaded for prediction.")
                contents = await file.read()
                await file.close()
                df = pd.read_csv(StringIO(contents.decode("utf-8")))

            # CASE 2: JSON body
            else:
                try:
                    data = await request.json()
                    if data:
                        logger.debug("JSON input received for prediction.")
                        df = pd.DataFrame([data])
                except Exception:
                    raise HTTPException(status_code=400, detail="Provide either a CSV file or JSON data.")

            if df is None:
                raise HTTPException(status_code=400, detail="Provide either file or JSON data.")

            # Validate schema
            try:
                validate_schema(df, EXPECTED_COLUMNS[:-1])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
            
            # Keep a copy of raw input for logging
            raw_inputs = df.copy(deep=True).to_dict(orient="records")

            df = df[EXPECTED_COLUMNS[:-1]]

            preds, probs = model.predict(model_input=df, return_proba=True, both=True)
            shap_values = model.explain_json(df)

            results = df.copy()
            results["prediction"] = preds
            results["probability"] = probs

            # Add background logging task
            background_tasks.add_task(log_predictions_task, raw_inputs, preds,probs)

            # Record Prometheus metrics
            duration = time.perf_counter() - start_time
            PREDICTION_COUNT.labels(model_name=MLFLOW_EXPERIMENT_NAME, version=version).inc()
            PREDICTION_LATENCY.labels(model_name=MLFLOW_EXPERIMENT_NAME, version=version).observe(duration)

            logger.info(f"Predictions generated for {len(df)} samples.")
            return JSONResponse({
                "role": APP_USERS.get(2),
                "results": results.to_dict(orient="records"),
                "shap_values": shap_values,
            })

        # AGENT ROLE
        elif current_user["role"] == APP_USERS.get(1):
            try:
                data = await request.json()
                logger.debug("JSON input received for prediction.")
            except Exception:
                raise HTTPException(status_code=400, detail="Agents must send JSON data only.")

            if not data:
                raise HTTPException(status_code=400, detail="Agents must send JSON data only.")

            df = pd.DataFrame([data])

            # Validate schema
            try:
                validate_schema(df, EXPECTED_COLUMNS[:-1])
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
            
            # Keep a copy of raw input for logging
            raw_inputs = df.copy(deep=True).to_dict(orient="records")

            df = df[EXPECTED_COLUMNS[:-1]]

            preds, probs = model.predict(model_input=df, return_proba=True, both=True)

            # Add background logging task
            background_tasks.add_task(log_predictions_task, raw_inputs, preds,probs)

            # Record Prometheus metrics
            duration = time.perf_counter() - start_time
            PREDICTION_COUNT.labels(model_name=MLFLOW_EXPERIMENT_NAME, version=version).inc()
            PREDICTION_LATENCY.labels(model_name=MLFLOW_EXPERIMENT_NAME, version=version).observe(duration)

            logger.info(f"Predictions generated for {len(df)} samples.")

            return JSONResponse({
                "role": APP_USERS.get(1),
                "prediction": int(preds[0]),
                "probability": float(probs[0]),
            })

        else:
            raise HTTPException(status_code=403, detail="Unauthorized role.")
    except HTTPException as e:
        version = app.state.version_folder.split("/")[-1]
        API_ERRORS.labels(endpoint=endpoint, method=method, status=e.status_code).inc()
        PREDICTION_ERRORS.labels(model_name=MLFLOW_EXPERIMENT_NAME, version=version).inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# 2. Feeedback Data (Only Admin)
@limiter.limit("10/minute")
@app.get("/feedback_data", dependencies=[Depends(verify_api_key)])
async def get_feedback_data(request: Request,limit: int = 50, payload: dict = Depends(require_role(APP_USERS.get(2)))):
    """
    Allows Admins to view recent model inference results.
    - Reads from the inference data CSV file.
    - Sorts by most recent timestamp.
    - Returns up to `limit` records (default = 50).
    - Useful for reviewing model decisions before giving feedback.
    """
    method = request.method
    endpoint = request.url.path
    try:
        logger.info("Admin requested feedback data retrieval.")
        df = read_csv_safe(INFERENCE_DATA_PATH)
        if df.empty:
            raise HTTPException(status_code=404, detail="No inference data found.")
        
        # Sort by timestamp and limit
        df = df.sort_values("timestamp", ascending=False).head(limit)
        records = df.to_dict(orient="records")

        logger.debug(f"Returning top {limit} recent inference records.")

        return records
    except HTTPException as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status=e.status_code).inc()
        logger.error(f"Feedback data retrieval failed: {e.detail}")
        raise

# 3. Feeedback(Human In The Loop) (Only Admin)
@limiter.limit("10/minute")
@app.post("/feedback", dependencies=[Depends(verify_api_key)])
async def feedback(request: Request,file: UploadFile = File(...),payload: dict = Depends(require_role(APP_USERS.get(2)))):
    """
    Allows Admins to upload human verified prediction data.
    - Uploads a CSV containing feedback (e.g., actual churn labels).
    - Merges it with existing inference data by `customerID`.
    - Updates existing records (no duplicates).
    - Saves merged data back to inference data.
    """
    method = request.method
    endpoint = request.url.path
    logger.info("Admin uploading feedback data file.")
    logger.debug(f"Uploaded file: {file.filename}")
    try:
        # Read uploaded CSV
        contents = await file.read()
        await file.close()
        updated_df = pd.read_csv(StringIO(contents.decode("utf-8")))

        if "customerID" not in updated_df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'customerID' column")

        # Read existing inference data
        existing_df = read_csv_safe(INFERENCE_DATA_PATH)

        # If no existing file, just save this one
        if existing_df.empty:
            updated_df.to_csv(INFERENCE_DATA_PATH, index=False)
            return {"message": "Feedback file saved as new inference data."}

        # Merge: keep all existing, update rows by customerID
        merged_df = pd.concat([existing_df, updated_df], ignore_index=True)
        merged_df.drop_duplicates(subset=["customerID"], keep="last", inplace=True)

        # Save back to file
        merged_df.to_csv(INFERENCE_DATA_PATH, index=False)

        logger.info("Feedback data merged successfully with existing inference records.")

        return {
            "message": "Feedback merged successfully with existing data.",
            "total_records": len(merged_df),
            "updated_records": len(updated_df)
        }

    except HTTPException as e:
        API_ERRORS.labels(endpoint=endpoint, method=method, status=e.status_code).inc()
        logger.error(f"Feedback failed: {e.detail}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {e}")

# 4. Upload Data (Only Admin)
@limiter.limit("10/minute")
@app.post("/upload_data", dependencies=[Depends(verify_api_key)])
async def upload_training_data(request: Request,file: UploadFile = File(...),payload: dict = Depends(require_role(APP_USERS.get(2)))):
    """
    Allows Admins to upload new or updated training data.
    - Accepts CSV file only.
    - Validates schema against expected columns.
    - Merges with existing training data.
    - Saves combined dataset to RAW_DATA_PATH.
    - Used before retraining to add new labeled data.
    """

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    try:
        # Read uploaded CSV
        contents = await file.read()
        await file.close()
        new_df = pd.read_csv(StringIO(contents.decode("utf-8")))

        if new_df.empty:
            raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")
        
        # Validate schema
        try:
            validate_schema(new_df, EXPECTED_COLUMNS)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

        # Load existing training data if it exists
        #merged_path = os.path.join(PROCESSED_DATA_DIR, processed_file_name)
        merged_path = RAW_DATA_PATH
        if os.path.exists(merged_path):
            existing_df = pd.read_csv(merged_path)

            # Merge and drop duplicates based on customerID
            combined_df = pd.concat([existing_df, new_df])
            combined_df.drop_duplicates(subset=["customerID"], keep="last", inplace=True)
        else:
            combined_df = new_df

        # Save merged CSV
        combined_df.to_csv(merged_path, index=False)

        return {
            "message": "Training data uploaded and merged successfully.",
            "file_saved_at": merged_path,
            "rows": len(combined_df),
            "columns": len(combined_df.columns)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload training data: {str(e)}")

# 5. Model Explainability (Only Admin)
@limiter.limit("10/minute")
@app.post("/model_explainability", dependencies=[Depends(verify_api_key)])
async def model_explainability(
    request: Request,
    file: Optional[UploadFile] = File(None),
    payload: dict = Depends(require_role(APP_USERS.get(2)))
):
    """
    Generates SHAP-based model explainability results.
    - Only accessible to Admins.
    - Accepts either CSV file or JSON input.
    - Validates and preprocesses data using stored preprocessing artifacts.
    - Returns SHAP values (feature importance per sample).
    - Helps interpret model predictions and feature influence.
    """
    if not hasattr(request.app.state, "model"):
        logger.warning("Model explainability requested but no model is loaded.")
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    logger.info("Explainability request received.")

    model = request.app.state.model
    df = None
    # CASE 1: File was uploaded
    if file is not None:
        contents = await file.read()
        await file.close()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # CASE 2: JSON body
    else:
        try:
            data = await request.json()
            if data:
                df = pd.DataFrame([data])
        except Exception:
            raise HTTPException(status_code=400, detail="Provide either a CSV file or JSON data.")

    if df is None:
        raise HTTPException(status_code=400, detail="Provide either file or JSON data.")

    # Validate schema
    try:
        validate_schema(df, EXPECTED_COLUMNS[:-1])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
    
    logger.debug(f"Input data shape: {df.shape}")
    df = df[EXPECTED_COLUMNS[:-1]]
    df = InferenceDatapreprocer(version_dir=app.state.version_folder).preprocess(df=df)


    shap_values = model.explain_json(df)
    results = df.copy()
    # results["prediction"] = preds
    # results["probability"] = probs

    logger.info("SHAP explanation generated successfully.")
    return JSONResponse({
        "role": APP_USERS.get(2),
        "results": results.to_dict(orient="records"),
        "shap_values": shap_values,
    })
  
# 6. Data Monitoring(Data distribution and Drift detection)  (Only Admin)
@limiter.limit("10/minute")
@app.get("/data_monitoring", dependencies=[Depends(verify_api_key)])
async def data_monitoring(request: Request, format: str = "html", window: int = None, payload: dict = Depends(require_role(APP_USERS.get(2)))):
    """
    Generates data quality and drift monitoring reports via Evidently Ai.
    - Only accessible to Admins.
    - Compares recent production data with training data.
    - Supports `DataDriftPreset` (drift detection) and `DataSummaryPreset` (summary stats).
    - Optional `window` parameter to analyze last N samples.
    - Returns either:
        - HTML report (for dashboards), or
        - JSON report (for APIs).
    """
    logger.info("Metric report generation requested.")
    if file_exist(INFERENCE_DATA_PATH) and file_exist(RAW_DATA_PATH):
        cur_data = pd.read_csv(INFERENCE_DATA_PATH)
        ref_data = pd.read_csv(RAW_DATA_PATH)
    
        if window is not None:
            cur_data = cur_data.sort_values(by=["timestamp"], ascending=True)
            n = min(window, len(cur_data))
            cur_data = cur_data.tail(n)

        if cur_data.shape[0]>=10:
            ref_data = InferenceDatapreprocer(version_dir=app.state.version_folder).data_cleaning(df=ref_data[EXPECTED_COLUMNS])
            cur_data = cur_data[EXPECTED_COLUMNS]

            report = Report(metrics=[DataDriftPreset()], include_tests=True)
            report = report.run(reference_data=ref_data, current_data=cur_data)
        else:
            if format.lower() == "html":
                html_content = "<html><body><h3>No enough production data for metrics evaluation (min 10 data points are required).</h3></body></html>"
                return HTMLResponse(content=html_content, status_code=400)
            else:
                return JSONResponse(content={"error": "No enough production data for metrics evaluation (min 10 data points are required)."}, status_code=400)

    elif file_exist(RAW_DATA_PATH):
        ref_data = pd.read_csv(RAW_DATA_PATH)
        report = Report(metrics=[DataSummaryPreset()])
        report = report.run(ref_data)
    else:
        if format.lower() == "html":
            html_content = "<html><body><h3>No data available for metrics.</h3></body></html>"
            return HTMLResponse(content=html_content, status_code=404)
        else:
            return JSONResponse(content={"error": "No data available"}, status_code=404)

    if format.lower() == "html":
        logger.debug("Returning HTML report.")
        return HTMLResponse(content=report.get_html_str(False))
    logger.debug("Returning JSON report.")
    return JSONResponse(content=report.dict())

# To be used by prometheus
@app.get("/metrics")
def metrics():
    data = generate_latest(REGISTRY)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

@app.post("/update_metrics",dependencies=[Depends(verify_api_key)])
def update_metrics(data: dict):
    """
    Centralized Prometheus metrics ingestion endpoint.
    Accepts JSON payloads from different monitoring subsystems:
    - drift: updates data drift metrics
    - pipeline_error: increments pipeline error counters
    - model_decay: updates model performance metrics
    """
    # Drift Metrics
    if data.get("from") == "drift":
        model_name = data.get("model_name", "unknown_model")
        drift_share = data.get("drift_share", 0.0)
        drift_count = data.get("drift_count", 0)

        DRIFT_SHARE_GAUGE.labels(model_name=model_name).set(drift_share)
        DRIFT_COUNT_GAUGE.labels(model_name=model_name).set(drift_count)

    # Pipeline Errors
    elif data.get("from") == "pipeline_error":
        pipeline_name = data.get("pipeline_name", "unknown_pipeline")
        PIPELINE_ERROR_COUNTER.labels(pipeline_name=pipeline_name).inc()

    # Model Decay Check
    elif data.get("from") == "model_decay":
        model_name = data.get("model_name", "unknown_model")
        version = str(data.get("version", "0"))
        model_type = data.get("model_type", "unknown_type")
        metrics = data.get("metric", {})

        # Update model performance gauges
        MODEL_ACCURACY_GAUGE.labels(model_name=model_name, model_type=model_type, version=version).set(metrics["accuracy"])
        MODEL_PRECISION_GAUGE.labels(model_name=model_name, model_type=model_type, version=version).set(metrics["precision"])
        MODEL_RECALL_GAUGE.labels(model_name=model_name, model_type=model_type, version=version).set(metrics["recall"])
        MODEL_ROC_AUC_GAUGE.labels(model_name=model_name, model_type=model_type, version=version).set(metrics["roc_auc"])
        MODEL_F1_GAUGE.labels(model_name=model_name, model_type=model_type, version=version).set(metrics["f1"])

    return {"status": "ok", "source": data.get("from")}
