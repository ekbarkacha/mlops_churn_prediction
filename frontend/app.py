"""
Module: frontend/app.py
=======================

Author: AIMS-AMMI STUDENT
Created: October/November 2025

Description:
------------
This module implements a production-ready admin portal for a customer churn
prediction system, using **Gradio** for the frontend interface and **MLflow**
for model handling.

It provides functionality for:
- User Authentication (register/login/logout/session handling)
- Single and Batch Churn Predictions
- SHAP-based feature importance visualization
- Feedback collection and submission
- Model monitoring (drift & performance reports)
- User management with role-based access (Admin vs Agent)
- Integration with MLflow, Prometheus, Grafana, and Airflow

Key Components:
---------------
1. **Authentication**
   - `register_user`: Registers a new user with username, password, and role.
   - `login`: Authenticates user and stores JWT token in session cookies.
   - `logout` / `auto_logout`: Clears session cookies, automatic session expiration.
   - `check_session` / `restore_session_on_load`: Auto-verify active session on page load.

2. **Prediction**
   - `validate_and_predict`: Validates input fields and performs single prediction.
   - `validate_csv` / `predict_csv`: Handles batch predictions from uploaded CSV files.
   - `generate_shap_output`: Generates SHAP explanations for feature importance.
   - `shap_plot_to_image`: Converts SHAP plots to images for display in the UI.

3. **Data Handling & Feedback**
   - `fetch_feedback_data`: Retrieves inference/feedback data and generates churn trend plots.
   - `load_inference_data`: Loads a limited number of inference records for review.
   - `submit_feedback`: Submits modified feedback back to the backend.

4. **Monitoring & Model Reporting**
   - `fetch_models_metrics`: Retrieves historical model metrics for performance monitoring.
   - `load_report_iframe`: Generates a full-page iframe HTML with a loading spinner for drift monitoring dashboards.
   - Sidebar buttons link to external monitoring tools: MLflow, Prometheus, Grafana, Airflow.

5. **User Management (Admin only)**
   - `fetch_users_from_api`: Fetches all users from the backend.
   - `approve_users`: Approves selected users in bulk.

6. **Frontend (Gradio Blocks)**
   - CSS styling for dark mode with sidebar navigation.
   - Role-based UI visibility: Admin vs Agent.
   - Tabs and groups for dashboard, predictions, batch predictions, model reports, model performance, user management, and feedback.
   - Automatic data loading after login based on user role.

7. **Utilities**
   - Constants: `BASE_URL`, `API_KEY`, `MLFLOW_URL`, `PROM_URL`, `GRAFANA_URL`, `AIRFLOW_URL`
   - `REQUIRED_COLUMNS` for CSV validation.
   - `session`: Persistent requests session for backend communication.
   - Threading for automatic session expiration.

Key Features:
-------------
- Role-based access control with dynamic UI updates.
- Real-time predictions with SHAP visualizations.
- CSV batch predictions with error handling.
- Integration with observability tools (MLflow, Prometheus, Grafana, Airflow).
- Secure API communication using session cookies and API keys.
- User-friendly dashboard with churn trends, model performance metrics, and feedback tables.
- Fully modular and extensible Gradio Blocks-based frontend.

"""

import gradio as gr
import requests
import os
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image
from dotenv import load_dotenv
import jwt, time, threading

load_dotenv()
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("API_KEY", "")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://127.0.0.1:5000")
PROM_URL = os.getenv("PROM_URL", "http://127.0.0.1:9090")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://127.0.0.1:3030")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://127.0.0.1:3080")

session = requests.Session()


REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]


def validate_and_predict(*args):
    field_names = [
        "Customer ID", "Gender", "Senior Citizen", "Partner", "Dependents",
        "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
        "Online Backup", "Device Protection", "Tech Support", "Streaming TV",
        "Streaming Movies", "Contract", "Paperless Billing", "Payment Method",
        "Monthly Charges", "Total Charges", "tenure"
    ]
    for name, value in zip(field_names, args):
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return f"❌ Field **'{name}'** is not filled or selected!", None, None, None, None

    input_data = {
        "customerID": args[0], "gender": args[1], "SeniorCitizen": args[2],
        "Partner": args[3], "Dependents": args[4], "PhoneService": args[5],
        "MultipleLines": args[6], "InternetService": args[7], "OnlineSecurity": args[8],
        "OnlineBackup": args[9], "DeviceProtection": args[10], "TechSupport": args[11],
        "StreamingTV": args[12], "StreamingMovies": args[13], "Contract": args[14],
        "PaperlessBilling": args[15], "PaymentMethod": args[16],
        "MonthlyCharges": args[17], "TotalCharges": args[18], "tenure": args[19]
    }
    return generate_shap_output(input_data)

def validate_csv(file):
    if file is None:
        return f"error: No file selected! Please upload a CSV file.", None, None, None
    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return f"error: Failed to read CSV: {str(e)}", None, None, None
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        return f"error: Missing columns: {', '.join(missing_cols)}", None, None, None
    return predict_csv(file)

def predict_single(data):
    response = session.post(f"{BASE_URL}/predict", json=data, headers={"x-api-key": API_KEY})
    if response.status_code == 200:
        return response.json()
    return {"error": response.text}

def predict_csv(file):
    files = {"file": open(file.name, "rb")}
    response = session.post(f"{BASE_URL}/predict", files=files, headers={"x-api-key": API_KEY})
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", [])
        df = pd.DataFrame(results)
        shap_values = data.get("shap_values")
        if shap_values:
            explanation = shap.Explanation(
                values=np.array(shap_values["raw"]["values"]),
                base_values=np.array(shap_values["raw"]["base_values"]),
                data=np.array([[data["results"][0][col] for col in shap_values["raw"]["columns"]]]),
                feature_names=shap_values["raw"]["columns"]
            )
            bar_img = shap_plot_to_image(shap.plots.bar, explanation)
            waterfall_img = shap_plot_to_image(shap.plots.waterfall, explanation[0])
        else:
            bar_img = None
            waterfall_img = None
        if "probability" in df.columns:
            df["probability"] = df["probability"].round(3)
        return None, df, bar_img, waterfall_img
    else:
        return f"error: {response.text}", None, None, None

def generate_shap_output(data):
    resp_json = predict_single(data)
    if "error" in resp_json:
        return resp_json["error"], None, None, None, None
    if resp_json.get("role") == "admin":
        shap_values = resp_json["shap_values"]
        explanation = shap.Explanation(
            values=np.array(shap_values["raw"]["values"]),
            base_values=np.array(shap_values["raw"]["base_values"]),
            data=np.array([[resp_json["results"][0][col] for col in shap_values["raw"]["columns"]]]),
            feature_names=shap_values["raw"]["columns"]
        )
        bar_img = shap_plot_to_image(shap.plots.bar, explanation)
        waterfall_img = shap_plot_to_image(shap.plots.waterfall, explanation[0])
        pred = resp_json["results"][0]["prediction"]
        prob = resp_json["results"][0]["probability"]
        risk = "Low" if prob < 0.33 else "Medium" if prob < 0.66 else "High"
        return None, f"{prob*100:.1f}%", risk, bar_img, waterfall_img
    else:
        pred = resp_json.get("prediction")
        prob = resp_json.get("probability")
        if prob is None:
            prob = 0.0
        risk = "Low" if prob < 0.33 else "Medium" if prob < 0.66 else "High"
        return None, f"{prob*100:.1f}%", risk, None, None

def shap_plot_to_image(plot_func, *args, **kwargs):
    plt.clf()
    plot_func(*args, show=False, **kwargs)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return np.array(Image.open(buf))

def fetch_models_metrics():
    try:
        response = session.get(f"{BASE_URL}/all_version_metrics", headers={"x-api-key": API_KEY})
        response.raise_for_status()
        data = response.json()
        rows = []
        for version, content in data.items():
            metrics = content.get("metrics", {})
            row = {"version": version, "run_id": content.get("run_id", None)}
            row.update(metrics)
            rows.append(row)
        if rows:
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame(columns=["version", "run_id"])
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])

def fetch_users_from_api():
    try:
        response = session.get(f"{BASE_URL}/users", headers={"x-api-key": API_KEY})
        response.raise_for_status()
        data = response.json()
        users = data.get("users", [])
        if not users:
            return pd.DataFrame(columns=["username", "role", "approved"])
        df = pd.DataFrame(users)
        return df
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}])

def approve_users(df):
    messages = []
    for _, row in df.iterrows():
        username = row["username"]
        approved = bool(row["approved"])
        if approved:
            try:
                resp = session.post(f"{BASE_URL}/approve_user", headers={"x-api-key": API_KEY}, json={"username": username, "approve": approved})
                resp.raise_for_status()
                messages.append(f"✅ {username} approved")
            except Exception as e:
                messages.append(f"❌ {username} failed: {str(e)}")
    if not messages:
        messages.append("No users were approved.")
    return "\n".join(messages)

def fetch_feedback_data(limit=50):
    try:
        response = session.get(f"{BASE_URL}/feedback_data?limit={limit}", headers={"x-api-key": API_KEY})
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        if df.empty or 'tenure' not in df.columns or 'Churn' not in df.columns:
            return "Dataset missing required columns (tenure, Churn).", None
        df['Churn_Flag'] = df['Churn'].map({'Yes': 1, 'No': 0})
        churn_trend = df.groupby('tenure')['Churn_Flag'].mean().reset_index()
        plt.figure(figsize=(10,6))
        sns.lineplot(data=churn_trend, x='tenure', y='Churn_Flag', marker='o', color='red')
        plt.title('Customer Churn Trend by Tenure')
        plt.xlabel('Tenure (Months)')
        plt.ylabel('Churn Rate')
        plt.grid(True)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        df = df.drop(columns=['Churn_Flag'])
        if 'probability' in df.columns:
            df['probability'] = df['probability'].round(3)
        return df, np.array(Image.open(buf))
    except Exception as e:
        return pd.DataFrame([{"error": str(e)}]), None

def load_inference_data(limit):
    MAX_LIMIT = 1000
    try:
        if limit is None or limit <= 0:
            return pd.DataFrame(), "Error: Limit must be a positive number."
        if limit > MAX_LIMIT:
            return pd.DataFrame(), f"Error: Limit cannot exceed {MAX_LIMIT} records."
        response = session.get(f"{BASE_URL}/feedback_data?limit={limit}", headers={"x-api-key": API_KEY})
        response.raise_for_status()
        records = response.json()
        if not records:
            return pd.DataFrame(), "No inference data found."
        df = pd.DataFrame(records)
        return df, f"Loaded {len(df)} recent inference records."
    except Exception as e:
        return pd.DataFrame(), f"Error loading data: {str(e)}"

def submit_feedback(df):
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        files = {"file": ("feedback.csv", csv_buffer, "text/csv")}
        headers = {"x-api-key": API_KEY}
        response = session.post(f"{BASE_URL}/feedback", files=files, headers=headers)
        response.raise_for_status()
        return response.json().get("message", "Submitted")
    except requests.exceptions.HTTPError as e:
        return f"Error submitting feedback: {e.response.status_code} {e.response.text}"
    except Exception as e:
        return f"Error submitting feedback: {str(e)}"
    
def load_report_iframe(window=50):
    token = session.cookies.get("access_token", "")
    report_url = f"{BASE_URL}/iframe_data_monitoring_proxy?api_key={API_KEY}&token={token}&format=html&window={window}"
    report_iframe = f'''
                    <style>
                    html, body {{
                    height: 100%;
                    margin: 0;
                    padding: 0;
                    }}
                    #iframe-container {{
                    position: relative;
                    width: 100%;
                    height: 100vh; /* Full viewport height */
                    }}
                    .spinner {{
                    border: 4px solid #2a2e36;
                    border-top: 4px solid #38bdf8;
                    border-radius: 50%;
                    width: 42px;
                    height: 42px;
                    animation: spin 1s linear infinite;
                    margin-bottom: 10px;
                    }}
                    @keyframes spin {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                    }}
                    #loader {{
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(17, 19, 23, 0.95);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    flex-direction: column;
                    color: #94a3b8;
                    font-size: 1em;
                    font-weight: 500;
                    z-index: 2;
                    border-radius: 8px;
                    transition: opacity 0.4s ease;
                    }}
                    </style>

                    <div id="iframe-container">
                    <!-- Loading overlay -->
                    <div id="loader">
                        <div class="spinner"></div>
                        Loading report... please wait
                    </div>

                    <!-- Iframe -->
                    <iframe src="{report_url}" width="100%" height="100%" 
                        style="border:1px solid #333; border-radius:8px; position: relative; z-index: 1;"
                        onload="this.previousElementSibling.style.opacity='0'; setTimeout(()=>this.previousElementSibling.style.display='none',400);">
                    </iframe>
                    </div>
                    '''
    return report_iframe

def register_user(username, password, role):
    if not username or not password:
        return "Please enter both username and password.", gr.update(visible=False), username, password
    response = session.post(f"{BASE_URL}/register", json={"username": username, "password": password, "role": role}, headers={"x-api-key": API_KEY})
    if response.status_code == 200:
        success_msg = f"User **{username}** successfully registered as **{role}**."
        return success_msg, gr.update(visible=True), "", ""
    elif response.status_code == 429:
        return f"Error: Too many requests, try again later", gr.update(visible=False), username, password
    else:
        try:
            return f"Error: {response.json().get('detail')}", gr.update(visible=False), username, password
        except:
            return "Error: Registration failed", gr.update(visible=False), username, password

def login(username, password):
    if not username or not password:
        return (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), f"Please enter both username and password.")
    response = session.post(f"{BASE_URL}/token", data={"username": username, "password": password}, headers={"x-api-key": API_KEY}, allow_redirects=False)
    if response.status_code == 200:
        token = response.cookies.get("access_token")
        if token:
            decoded = jwt.decode(token, options={"verify_signature": False})
            exp = decoded.get("exp")
            remaining = exp - int(time.time())
            threading.Thread(target=auto_logout, args=(remaining,), daemon=True).start()
            return (gr.update(value=""), gr.update(value=""), gr.update(visible=False), gr.update(visible=True), "")
    elif response.status_code == 429:
        return (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), f"Error: Too many requests, try again later")
    else:
        try:
            return (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), f"Error: {response.json().get('detail')}")
        except:
            return (gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), f"Error: Login failed")

def check_session():
    try:
        r = session.get(f"{BASE_URL}/verify-session", headers={"x-api-key": API_KEY})
        if r.status_code == 200:
            user = r.json()
            return f"Welcome back {user['username']} ({user['role']})", gr.update(visible=False), gr.update(visible=True)
        return "Session expired or not found. Please log in.", gr.update(visible=True), gr.update(visible=False)
    except Exception as e:
        return f"Error verifying session: {e}", gr.update(visible=True), gr.update(visible=False)

def auto_logout(delay):
    time.sleep(delay)
    session.cookies.clear()
    print("Session expired automatically.")

def logout():
    session.cookies.clear()
    return gr.update(visible=True), gr.update(visible=False)


# CSS
dark_cs = """
body { background-color: #0f1115 !important; color: #e5e5e5 !important; }
.sidebar { background: #111317; padding: 20px; border-right: 1px solid rgba(255,255,255,0.05); min-height: 95vh; }
.sidebar-button { width: 100%; text-align: left; margin-bottom: 6px; padding: 10px 12px; border-radius: 8px; background: #1c1f26; color: #fff; border: none; cursor: pointer; }
.sidebar-button:hover { background: #2a2e36; }
.section-title { color: #9ca3af; font-weight: 600; margin-top: 14px; margin-bottom: 6px; }
.brand { color: #fff; font-size: 18px; font-weight: bold; margin-bottom: 8px; }
"""

dark_css = """
/* ====== Global Layout ====== */
body {
    background-color: #0f1115 !important;
    color: #e5e5e5 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ====== Sidebar Styling ====== */
.sidebar {
    background: #111317;
    padding: 20px 16px;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
    min-height: 95vh;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

/* ====== Buttons in Sidebar ====== */
.sidebar-button {
    width: 100%;
    text-align: left;
    font-weight: 500;
    background-color: #1c1f26 !important;
    color: #e2e8f0 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    transition: all 0.2s ease-in-out;
    cursor: pointer;
}

.sidebar-button:hover {
    background-color: #2a2e36 !important;
    color: #f8fafc !important;
    transform: translateX(3px);
}

.sidebar-button:focus {
    outline: 2px solid #38bdf8 !important;
    outline-offset: 1px;
}

/* ====== Text and Section Headers ====== */
.section-title {
    color: #9ca3af;
    font-weight: 600;
    margin-top: 18px;
    margin-bottom: 8px;
    text-transform: uppercase;
    font-size: 0.8em;
    letter-spacing: 0.6px;
}

/* ====== Branding / Footer ====== */
.brand {
    color: #ffffff;
    font-size: 1.15em;
    font-weight: 700;
    text-align: center;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
}

.small-muted {
    color: #94a3b8;
    font-size: 0.8em;
    text-align: center;
}

hr {
    border: 0;
    height: 1px;
    background: #2e3440;
    margin: 18px 0;
}
"""


with gr.Blocks(css=dark_css, title="Churn Prediction — Admin Portal", theme=gr.themes.Base()) as demo:
    session_status = gr.Markdown(value="") 

    # AUTH group 
    with gr.Group(visible=True) as auth_group:
        with gr.Column(scale=0.6, min_width=500):
            gr.Markdown("## User Authentication Portal")
            with gr.Tab("Register"):
                reg_username = gr.Textbox(label="Username")
                reg_password = gr.Textbox(label="Password", type="password")
                reg_role = gr.Dropdown(["Agent", "Admin"], value="Agent", label="Role")
                register_btn = gr.Button("Register", variant="primary")
                register_msg = gr.Markdown()
                register_btn.click(register_user, inputs=[reg_username, reg_password, reg_role], outputs=[register_msg, gr.Column(), reg_username, reg_password])
            with gr.Tab("Login"):
                username_in = gr.Textbox(label="Username")
                password_in = gr.Textbox(label="Password", type="password")
                login_btn = gr.Button("Login", variant="primary")
                login_msg = gr.Markdown()

    # MAIN APP
    with gr.Row(visible=False) as main_app:

        with gr.Column(scale=0.24,min_width=240,elem_classes="sidebar",):
            gr.HTML("""
                <div style="
                    text-align:center;
                    font-size:1.3em;
                    font-weight:700;
                    color:#e2e8f0;
                    letter-spacing:0.5px;
                    margin-top:12px;
                ">
                    📡 Churn Prediction Console
                </div>
            """)

            gr.Markdown(
                """
                <div style="
                    text-align:center;
                    font-size:0.82em;
                    color:#94a3b8;
                    margin-bottom:22px;
                ">
                    Secure · Role-Based · Production-Ready
                </div>
                """,
                elem_classes="small-muted"
            )

            btn_dashboard = gr.Button("Dashboard", elem_classes="sidebar-button", visible=False)
            btn_prediction = gr.Button("Prediction", elem_classes="sidebar-button", visible=True)
            btn_batch = gr.Button("Batch Prediction (CSV)", elem_classes="sidebar-button", visible=False)
            btn_model_report = gr.Button("Data Drift Monitoring", elem_classes="sidebar-button", visible=False)
            btn_model_perf = gr.Button("Model Performance", elem_classes="sidebar-button", visible=False)
            btn_user_mgmt = gr.Button("User Management", elem_classes="sidebar-button", visible=False)
            btn_feedback = gr.Button("Feedback", elem_classes="sidebar-button", visible=False)

            with gr.Column(visible=False) as tab_link:
                gr.HTML("<hr style='border: 0; height: 1px; background: #334155; margin: 18px 0;'>")
                gr.Markdown(
                    "<div style='text-align:center; font-size:0.85em; color:#cbd5e1; margin-bottom:6px;'>🔎 Monitoring & Tools</div>"
                )

                gr.HTML(f"""
                <div style="display:flex; flex-direction:column; gap:8px;">
                    <a href="{MLFLOW_URL}" target="_blank"><button class="sidebar-button">🧪 MLflow</button></a>
                    <a href="{PROM_URL}" target="_blank"><button class="sidebar-button">📡 Prometheus</button></a>
                    <a href="{GRAFANA_URL}" target="_blank"><button class="sidebar-button">📊 Grafana</button></a>
                    <a href="{AIRFLOW_URL}" target="_blank"><button class="sidebar-button">💨 Airflow</button></a>
                </div>
                """)


            gr.HTML("<hr style='border: 0; height: 1px; background: #334155; margin: 18px 0;'>")
            btn_logout = gr.Button(" Logout", variant="secondary", visible=True)
            gr.Markdown(
                """
                <div style="
                    text-align:center;
                    font-size:0.75em;
                    color:#64748b;
                    margin-top:6px;
                ">
                    Version 1.0 — Dark Mode
                </div>
                """
            )


        # MAIN content area
        with gr.Column(scale=0.76):
            
            # MAIN APP (hidden until logged in)
            with gr.Group(visible=True) as main_ap:
                # Dashboard view
                with gr.Group(visible=False) as dashboard_view:
                    gr.Markdown("### Dashboard")
                    churn_trend_img = gr.Image(label="Churn Trend", interactive=False)
                    sample_table = gr.Dataframe(pd.DataFrame(), label="Sample Customer Data")

                # Prediction view (visible to both roles)
                with gr.Group(visible=False) as prediction_view:
                    gr.Markdown("### Single Prediction (Manual Input)")
                    with gr.Row():
                        customerID = gr.Textbox(label="Customer ID")
                        gender = gr.Dropdown(["Male","Female"], label="Gender")
                        senior = gr.Radio([0,1], label="Senior Citizen")
                        partner = gr.Radio(["Yes","No"], label="Partner")
                        dependents = gr.Radio(["Yes","No"], label="Dependents")
                    with gr.Row():
                        phone = gr.Radio(["Yes","No"], label="Phone Service")
                        multiple_lines = gr.Radio(["Yes","No","No phone service"], label="Multiple Lines")
                        internet = gr.Dropdown(["DSL","Fiber optic","No"], label="Internet Service")
                        online_security = gr.Radio(["Yes","No","No internet service"], label="Online Security")
                        online_backup = gr.Radio(["Yes","No","No internet service"], label="Online Backup")
                    with gr.Row():
                        device_protection = gr.Radio(["Yes","No","No internet service"], label="Device Protection")
                        tech_support = gr.Radio(["Yes","No","No internet service"], label="Tech Support")
                        streaming_tv = gr.Radio(["Yes","No","No internet service"], label="Streaming TV")
                        streaming_movies = gr.Radio(["Yes","No","No internet service"], label="Streaming Movies")
                    with gr.Row():
                        contract = gr.Dropdown(["Month-to-month","One year","Two year"], label="Contract")
                        paperless = gr.Radio(["Yes","No"], label="Paperless Billing")
                        payment = gr.Dropdown(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], label="Payment Method")
                        monthly_charges = gr.Number(label="Monthly Charges")
                        total_charges = gr.Number(label="Total Charges")
                        tenure = gr.Number(label="Tenure")
                    with gr.Row():
                        predict_btn = gr.Button("Predict", variant="primary")
                        pred_error = gr.Markdown()
                    churn_prob = gr.Textbox(label="Predicted Churn Probability (%)")
                    risk_level = gr.Textbox(label="Risk Level")

                    with gr.Group(visible=False) as sharp_img:
                        with gr.Column(scale=2):
                            with gr.Row():
                                bar_plot = gr.Image(label="Feature Importance (SHAP Bar Plot)")
                                waterfall_plot = gr.Image(label="Prediction Breakdown (Waterfall Plot)")

                    
                    predict_btn.click(
                        validate_and_predict,
                        inputs=[customerID, gender, senior, partner, dependents, phone, multiple_lines,
                                internet, online_security, online_backup, device_protection, tech_support,
                                streaming_tv, streaming_movies, contract, paperless, payment,
                                monthly_charges, total_charges, tenure],
                        outputs=[pred_error, churn_prob, risk_level, bar_plot, waterfall_plot]
                    )

                # Batch predictive (admin)
                with gr.Group(visible=False) as batch_view:
                    gr.Markdown("### Batch Prediction (Admin)")
                    csv_file = gr.File(label="Upload CSV", file_count="single")
                    run_csv_btn = gr.Button("Run Batch Prediction", variant="primary")
                    batch_err = gr.Markdown()
                    batch_results = gr.Dataframe()

                    with gr.Group(visible=True):
                        with gr.Column(scale=2):
                            with gr.Row():
                                csv_bar_plot = gr.Image(label="Feature Importance (SHAP Bar Plot)")
                                csv_waterfall_plot = gr.Image(label="Prediction Breakdown (Waterfall Plot)")

                    run_csv_btn.click(validate_csv, inputs=csv_file, outputs=[batch_err, batch_results, csv_bar_plot, csv_waterfall_plot])


                # Model report (admin)
                with gr.Group(visible=False) as model_report_view:
                    gr.Markdown("### Data Drift Monitoring Report")
    
                    window_input = gr.Slider(label="Window Size", minimum=1, maximum=5000, value=50, step=1)
                    load_report_btn = gr.Button("Load Report")
                    report_html = gr.HTML("<div>Loading report...</div>")
                    load_report_btn.click(fn=load_report_iframe, inputs=window_input, outputs=report_html)

                    
                # Model performance (admin)
                with gr.Group(visible=False) as model_perf_view:
                    gr.Markdown("### Model Performance")
                    metrics_table = gr.Dataframe()

                # User management (admin)
                with gr.Group(visible=False) as user_mgmt_view:
                    gr.Markdown("### User Management")
                    user_df = gr.Dataframe(value=pd.DataFrame(columns=["username","role","approved"]), interactive=True, datatype=["str","str","bool"], label="Users")
                    approve_msg = gr.Markdown()
                    refresh_users_btn = gr.Button("Refresh Users", variant="secondary")
                    save_users_btn = gr.Button("Save Changes", variant="primary")
                    refresh_users_btn.click(fn=fetch_users_from_api, outputs=user_df)
                    save_users_btn.click(fn=approve_users, inputs=user_df, outputs=approve_msg)

                # Feedback (admin)
                with gr.Group(visible=False) as feedback_view:
                    gr.Markdown("### Human-in-the-loop Feedback")
                    limit_input = gr.Number(label="Records to load", value=50, precision=0)
                    load_feedback_btn = gr.Button("Load Inference Data")
                    submit_feedback_btn = gr.Button("Submit Modified Feedback")
                    status_text = gr.Textbox()
                    data_table = gr.Dataframe(interactive=True)
                    load_feedback_btn.click(fn=load_inference_data, inputs=limit_input, outputs=[data_table, status_text])
                    submit_feedback_btn.click(fn=submit_feedback, inputs=data_table, outputs=status_text)

    # ----- Helper to auto-load data after login -----
    def autoload_for_role(role):
        """
        Return a tuple of (df_sample_table, churn_img, metrics_df, users_df, feedback_df, report_iframe_html)
        Any not-applicable values can be None.
        """
       
        df_sample, churn_img = pd.DataFrame(), None
        metrics_df = pd.DataFrame()
        users_df = pd.DataFrame()
        feedback_df = pd.DataFrame()
         # Dashboard (churn trend + sample)
        if role == "admin":
            try:
                df_sample, churn_img = fetch_feedback_data(50)
            except Exception:
                df_sample, churn_img = pd.DataFrame(), None

            # metrics
            try:
                metrics_df = fetch_models_metrics()
            except Exception:
                metrics_df = pd.DataFrame()

            # users
            if role == "admin":
                try:
                    users_df = fetch_users_from_api()
                except Exception:
                    users_df = pd.DataFrame()

            # feedback table (inference)
            if role == "admin":
                try:
                    feedback_df, _ = load_inference_data(50)
                except Exception:
                    feedback_df = pd.DataFrame()

        # model report iframe only for admin
        report_iframe = ""
        if role == "admin":
            report_iframe = load_report_iframe()
            
        return df_sample, churn_img, metrics_df, users_df, feedback_df, report_iframe

    # ----- Login handler: authenticate, verify role, auto-load and set UI visibility -----
    def handle_login_click(username_val, password_val):
        # Basic validation
        if not username_val or not password_val:
            return (
                gr.update(value="Please enter both username and password."),  # login_msg
                gr.update(visible=True),  # auth_group remains visible
                gr.update(visible=False), # main_app hidden
                # placeholders so outputs align (see mapping below)
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),gr.update(), gr.update(), gr.update(),gr.update(),gr.update(),gr.update(),gr.update(),gr.update()
            )
            

        # POST to /token
        try:
            resp = session.post(f"{BASE_URL}/token", data={"username": username_val, "password": password_val}, headers={"x-api-key": API_KEY}, allow_redirects=False)
        except Exception as e:
            return (gr.update(value=f"Error contacting auth endpoint: {e}"), gr.update(visible=True), gr.update(visible=False),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),gr.update(), gr.update(), gr.update(),gr.update(),gr.update(),gr.update(),gr.update(),gr.update())

        if resp.status_code != 200:
            try:
                reason = resp.json().get("detail", resp.text)
            except Exception:
                reason = resp.text
            return (gr.update(value=f"Login failed: {reason}"), gr.update(visible=True), gr.update(visible=False),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),gr.update(), gr.update(), gr.update(),gr.update(),gr.update(),gr.update(),gr.update(),gr.update())

        # success: token stored in session cookies by backend
        token = session.cookies.get("access_token")
        if token:
            try:
                decoded = jwt.decode(token, options={"verify_signature": False})
                exp = decoded.get("exp", None)
                if exp:
                    remaining = exp - int(time.time())
                    if remaining > 0:
                        threading.Thread(target=auto_logout, args=(remaining,), daemon=True).start()
            except Exception:
                pass

        # verify session to get user info (username, role)
        try:
            r = session.get(f"{BASE_URL}/verify-session", headers={"x-api-key": API_KEY})
        except Exception as e:
            return (gr.update(value=f"Login succeeded but session verify failed: {e}"), gr.update(visible=True), gr.update(visible=False),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),gr.update(), gr.update(), gr.update(),gr.update(),gr.update(),gr.update(),gr.update(),gr.update())

        if r.status_code != 200:
            return (gr.update(value="Login succeeded but could not verify session."), gr.update(visible=True), gr.update(visible=False),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),gr.update(), gr.update(), gr.update(),gr.update(),gr.update(),gr.update(),gr.update(),gr.update())

        user = r.json()
        role = user.get("role", "agent").lower()
        username = user.get("username", "user")

        # autoload data for role
        df_sample, churn_img, metrics_df, users_df, feedback_df, report_iframe = autoload_for_role(role)

        # build admin-visible flags
        admin_vis = role == "admin"
        not_admin_vis = role != "admin"


        # prepare return outputs (must match outputs list in login_btn.click mapping below)
        return (
            gr.update(value=""),        # login_msg
            gr.update(visible=False),  # hide auth_group
            gr.update(visible=True),   # show main_app
            # dashboard outputs:
            gr.update(value=df_sample),    # sample_table
            gr.update(value=churn_img),    # churn_trend_img
            # metrics
            gr.update(value=metrics_df),   # metrics_table
            # users
            gr.update(value=users_df),     # user_df
            # feedback
            gr.update(value=feedback_df),  # data_table
            # model report html
            gr.update(value=report_iframe),# report_html
            # toggle visibility for admin-only items:
            gr.update(visible=admin_vis),  # btn_dashboard
            gr.update(visible=True),       # btn_prediction (always shown)
            gr.update(visible=admin_vis),  # btn_batch
            gr.update(visible=admin_vis),  # btn_model_report
            gr.update(visible=admin_vis),  # btn_model_perf
            gr.update(visible=admin_vis),  # btn_user_mgmt
            gr.update(visible=admin_vis),   # btn_feedback
            gr.update(visible=admin_vis),   # dashboard view
            gr.update(visible=not_admin_vis),   # prediction view
            gr.update(visible=admin_vis), # links layout
            gr.update(visible=admin_vis)  # shap images
            

        )

    # wire login button outputs: many outputs must match the tuple above
    login_btn.click(
        fn=handle_login_click,
        inputs=[username_in, password_in],
        outputs=[
            login_msg,     # message
            auth_group,    # hide auth
            main_app,      # show main app
            sample_table,  # sample_table
            churn_trend_img, # churn image
            metrics_table, # metrics_table
            user_df,       # user dataframe
            data_table,    # feedback data_table
            report_html,   # report_html
            btn_dashboard, # visibility toggles for sidebar buttons
            btn_prediction,
            btn_batch,
            btn_model_report,
            btn_model_perf,
            btn_user_mgmt,
            btn_feedback,
            dashboard_view,
            prediction_view,
            tab_link,
            sharp_img
        ]
    )

    # Logout mapping
    def handle_logout_click():
        session.cookies.clear()
        return gr.update(visible=True), gr.update(visible=False), gr.update(value="🔒 Signed out")
    btn_logout.click(fn=handle_logout_click, outputs=[auth_group, main_app, session_status])

    # Sidebar navigation handlers: show the selected view, hide others.
    def show_dashboard():
        return (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    def show_prediction():
        return (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    def show_batch():
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    def show_report():
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))
    def show_perf():
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False))
    def show_users():
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=False))
    def show_infrence():
        return (gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True))

    btn_dashboard.click(fn=show_dashboard, outputs=[dashboard_view, prediction_view, batch_view, model_report_view, model_perf_view, user_mgmt_view,feedback_view])
    btn_prediction.click(fn=show_prediction, outputs=[dashboard_view, prediction_view, batch_view, model_report_view, model_perf_view, user_mgmt_view,feedback_view])
    btn_batch.click(fn=show_batch, outputs=[dashboard_view, prediction_view, batch_view, model_report_view, model_perf_view, user_mgmt_view,feedback_view])
    btn_model_report.click(fn=show_report, outputs=[dashboard_view, prediction_view, batch_view, model_report_view, model_perf_view, user_mgmt_view,feedback_view])
    btn_model_perf.click(fn=show_perf, outputs=[dashboard_view, prediction_view, batch_view, model_report_view, model_perf_view, user_mgmt_view,feedback_view])
    btn_user_mgmt.click(fn=show_users, outputs=[dashboard_view, prediction_view, batch_view, model_report_view, model_perf_view, user_mgmt_view,feedback_view])
    
    btn_feedback.click(fn=show_infrence, outputs=[dashboard_view, prediction_view, batch_view, model_report_view, model_perf_view, user_mgmt_view,feedback_view])


    # AUTO SESSION RESTORE
    def restore_session_on_load():
        """Auto-check session when the page loads or refreshes."""
        try:
            r = session.get(f"{BASE_URL}/verify-session", headers={"x-api-key": API_KEY})
            if r.status_code == 200:
                user = r.json()
                role = user.get("role", "agent").lower()
                username = user.get("username", "user")
                df_sample, churn_img, metrics_df, users_df, feedback_df, report_iframe = autoload_for_role(role)
                admin_vis = role == "admin"
                not_admin_vis = role != "admin"
                return (
                    gr.update(value=f""), #Welcome back {username} ({role})
                    gr.update(visible=False),   # auth_group
                    gr.update(visible=True),    # main_app
                    gr.update(value=df_sample),
                    gr.update(value=churn_img),
                    gr.update(value=metrics_df),
                    gr.update(value=users_df),
                    gr.update(value=feedback_df),
                    gr.update(value=report_iframe),
                    gr.update(visible=admin_vis),  # dashboard button
                    gr.update(visible=True),       # prediction button
                    gr.update(visible=admin_vis),  # batch button
                    gr.update(visible=admin_vis),  # model report
                    gr.update(visible=admin_vis),  # model perf
                    gr.update(visible=admin_vis),  # user mgmt
                    gr.update(visible=admin_vis),  # feedback
                    gr.update(visible=admin_vis),  # dashboard view
                    gr.update(visible=not_admin_vis),  # prediction view
                    gr.update(visible=admin_vis),  # tab link
                    gr.update(visible=admin_vis),  # shap img
                )
            else:
                # session invalid
                return (
                    gr.update(value=""), #Session expired. Please log in again.
                    gr.update(visible=True),
                    gr.update(visible=False),
                    *[gr.update() for _ in range(17)]
                )
        except Exception as e:
            return (
                gr.update(value=f"Error verifying session: {e}"),
                gr.update(visible=True),
                gr.update(visible=False),
                *[gr.update() for _ in range(17)]
            )

    demo.load(
        fn=restore_session_on_load,
        inputs=[],
        outputs=[
            login_msg, auth_group, main_app,
            sample_table, churn_trend_img, metrics_table, user_df, data_table, report_html,
            btn_dashboard, btn_prediction, btn_batch, btn_model_report, btn_model_perf,
            btn_user_mgmt, btn_feedback, dashboard_view, prediction_view, tab_link, sharp_img
        ]
    )


# Launch
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7862, debug=False, share=False)

