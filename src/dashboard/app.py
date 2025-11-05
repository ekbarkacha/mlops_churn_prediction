"""
Customer Churn Prediction Dashboard
Streamlit Interactive Dashboard
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.dashboard.api_client import ChurnAPIClient
from src.dashboard.components import (
    render_header,
    render_metrics_cards,
    render_prediction_form,
    render_batch_upload,
    render_model_info,
    render_analytics
)

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ff4444;
        color: white;
    }
    .medium-risk {
        background-color: #ffaa00;
        color: white;
    }
    .low-risk {
        background-color: #00C851;
        color: white;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_api_client():
    return ChurnAPIClient(
        base_url="http://127.0.0.1:8000",
        api_key="dev_key_12345"
    )

api_client = get_api_client()

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔮 Single Prediction", "📁 Batch Prediction", "📈 Model Performance", "📊 Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Customer Churn Prediction Dashboard**
    
    Predict customer churn probability using ML models.
    
    **Features:**
    - Single customer prediction
    - Batch CSV upload
    - Model performance metrics
    - Interactive analytics
    """
)

# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<h1 class="main-header">📊 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Churn Prediction System
    
    This dashboard provides an interactive interface to predict customer churn using advanced machine learning models.
    
    **Key Features:**
    - 🎯 **Single Prediction:** Analyze individual customer profiles
    - 📁 **Batch Processing:** Upload CSV files for bulk predictions
    - 📈 **Performance Metrics:** View model accuracy and statistics
    - 📊 **Analytics:** Explore data insights and trends
    """)
    
    # Check API health
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        health = api_client.health_check()
        
        with col1:
            st.metric("API Status", "🟢 Online" if health.get("status") == "healthy" else "🔴 Offline")
        
        with col2:
            st.metric("Model Status", "✅ Loaded" if health.get("model_loaded") else "❌ Not Loaded")
        
        with col3:
            model_info = health.get("model_info", {})
            st.metric("F1 Score", f"{model_info.get('f1_score', 0):.2%}")
        
        with col4:
            st.metric("ROC AUC", f"{model_info.get('roc_auc', 0):.2%}")
    
    except Exception as e:
        st.error(f"⚠️ Cannot connect to API: {e}")
        st.info("Make sure the API is running: `uvicorn src.deployment.api:app --reload`")
    
    st.markdown("---")
    
    # Quick stats
    st.subheader("📊 Quick Stats")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Information:**
        - Algorithm: XGBoost
        - Features: 13 basic features
        - Training: SMOTE balanced data
        - Version: Production v1.0
        """)
    
    with col2:
        st.markdown("""
        **Risk Levels:**
        - 🔴 **HIGH:** Probability ≥ 70%
        - 🟡 **MEDIUM:** 40% ≤ Probability < 70%
        - 🟢 **LOW:** Probability < 40%
        """)

# ============================================================================
# PAGE 2: SINGLE PREDICTION
# ============================================================================
elif page == "🔮 Single Prediction":
    st.markdown('<h1 class="main-header">🔮 Single Customer Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("### Enter Customer Information")
    
    # Form layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Demographics")
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        partner = st.selectbox("Has Partner", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        dependents = st.selectbox("Has Dependents", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        
        st.subheader("📊 Account Info")
        tenure = st.slider("Tenure (scaled 0-1)", 0.0, 1.0, 0.5, 0.01, 
                          help="0 = New customer, 1 = Long-term customer (72 months)")
        monthly_charges = st.slider("Monthly Charges (scaled 0-1)", 0.0, 1.0, 0.5, 0.01)
        total_charges = st.slider("Total Charges (scaled 0-1)", 0.0, 1.0, 0.5, 0.01)
    
    with col2:
        st.subheader("🛡️ Services")
        online_security = st.selectbox("Online Security", [0, 1, 2], 
                                       format_func=lambda x: ["No", "Yes", "No Service"][x])
        online_backup = st.selectbox("Online Backup", [0, 1, 2],
                                     format_func=lambda x: ["No", "Yes", "No Service"][x])
        device_protection = st.selectbox("Device Protection", [0, 1, 2],
                                        format_func=lambda x: ["No", "Yes", "No Service"][x])
        tech_support = st.selectbox("Tech Support", [0, 1, 2],
                                    format_func=lambda x: ["No", "Yes", "No Service"][x])
        
        st.subheader("📋 Contract")
        contract = st.selectbox("Contract Type", [0, 1, 2],
                               format_func=lambda x: ["Month-to-month", "One year", "Two year"][x])
        paperless_billing = st.selectbox("Paperless Billing", [0, 1],
                                        format_func=lambda x: "Yes" if x == 1 else "No")
        payment_method = st.selectbox("Payment Method", [0, 1, 2, 3],
                                     format_func=lambda x: ["Electronic check", "Mailed check", 
                                                           "Bank transfer", "Credit card"][x])
    
    # Predict button
    if st.button("🎯 Predict Churn", type="primary"):
        with st.spinner("Analyzing customer profile..."):
            try:
                # Prepare data
                customer_data = {
                    "SeniorCitizen": senior_citizen,
                    "Partner": partner,
                    "Dependents": dependents,
                    "tenure": tenure,
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                    "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup,
                    "DeviceProtection": device_protection,
                    "TechSupport": tech_support,
                    "Contract": contract,
                    "PaperlessBilling": paperless_billing,
                    "PaymentMethod": payment_method
                }
                
                # Make prediction
                result = api_client.predict([customer_data])
                
                if result and "predictions" in result:
                    pred = result["predictions"][0]
                    
                    # Display result
                    st.markdown("---")
                    st.subheader("🎯 Prediction Result")
                    
                    # Risk box
                    risk_level = pred.get("risk_level", "LOW")
                    risk_class = f"{risk_level.lower()}-risk"
                    
                    st.markdown(f"""
                    <div class="prediction-box {risk_class}">
                        <h2>Prediction: {pred.get('prediction', 'Unknown')}</h2>
                        <h3>Churn Probability: {pred.get('churn_probability', 0):.1%}</h3>
                        <h3>Risk Level: {risk_level}</h3>
                        <h4>Confidence: {pred.get('confidence', 0):.1%}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=pred.get('churn_probability', 0) * 100,
                        title={'text': "Churn Risk %"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Recommendations
                    st.subheader("💡 Recommendations")
                    
                    if risk_level == "HIGH":
                        st.error("""
                        **⚠️ HIGH RISK CUSTOMER - IMMEDIATE ACTION REQUIRED**
                        
                        - 📞 Contact customer within 24 hours
                        - 🎁 Offer retention incentives (discount, upgrade)
                        - 🤝 Schedule personalized consultation
                        - 📊 Review service satisfaction
                        """)
                    elif risk_level == "MEDIUM":
                        st.warning("""
                        **⚠️ MEDIUM RISK - PROACTIVE MEASURES RECOMMENDED**
                        
                        - 📧 Send personalized engagement email
                        - 🎯 Offer relevant service upgrades
                        - 📞 Conduct satisfaction survey
                        - 🔔 Monitor account activity
                        """)
                    else:
                        st.success("""
                        **✅ LOW RISK - MAINTAIN ENGAGEMENT**
                        
                        - ✉️ Continue regular communications
                        - 🌟 Consider loyalty rewards program
                        - 📱 Promote new features/services
                        - 😊 Request feedback for improvements
                        """)
                
                else:
                    st.error("Failed to get prediction")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# ============================================================================
# PAGE 3: BATCH PREDICTION
# ============================================================================
elif page == "📁 Batch Prediction":
    st.markdown('<h1 class="main-header">📁 Batch Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Upload CSV File for Bulk Predictions
    
    Upload a CSV file containing multiple customer records to get predictions for all of them at once.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! {len(df)} customers found.")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Predict button
            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner(f"Processing {len(df)} customers..."):
                    try:
                        # Convert to list of dicts
                        customers = df.to_dict('records')
                        
                        # Make predictions
                        result = api_client.predict(customers)
                        
                        if result and "predictions" in result:
                            predictions = result["predictions"]
                            
                            # Add predictions to dataframe
                            df['Prediction'] = [p['prediction'] for p in predictions]
                            df['Churn_Probability'] = [p['churn_probability'] for p in predictions]
                            df['Risk_Level'] = [p['risk_level'] for p in predictions]
                            df['Confidence'] = [p['confidence'] for p in predictions]
                            
                            st.success("✅ Predictions completed!")
                            
                            # Summary stats
                            st.subheader("📊 Summary Statistics")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                churn_count = len(df[df['Prediction'] == 'Churn'])
                                st.metric("Predicted Churns", churn_count, 
                                         f"{churn_count/len(df)*100:.1f}%")
                            
                            with col2:
                                high_risk = len(df[df['Risk_Level'] == 'HIGH'])
                                st.metric("High Risk", high_risk,
                                         f"{high_risk/len(df)*100:.1f}%")
                            
                            with col3:
                                avg_prob = df['Churn_Probability'].mean()
                                st.metric("Avg Churn Probability", f"{avg_prob:.1%}")
                            
                            with col4:
                                avg_conf = df['Confidence'].mean()
                                st.metric("Avg Confidence", f"{avg_conf:.1%}")
                            
                            # Visualizations
                            st.subheader("📈 Visualizations")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Risk distribution
                                risk_counts = df['Risk_Level'].value_counts()
                                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                                           title="Risk Level Distribution",
                                           color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Probability distribution
                                fig = px.histogram(df, x='Churn_Probability', nbins=20,
                                                 title="Churn Probability Distribution",
                                                 labels={'Churn_Probability': 'Churn Probability'})
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Results table
                            st.subheader("📋 Detailed Results")
                            st.dataframe(df, use_container_width=True)
                            
                            # Download button
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv,
                                file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
    
    else:
        # Show example format
        st.info("💡 **CSV Format Example:**")
        
        example_df = pd.DataFrame({
            'SeniorCitizen': [0, 1],
            'Partner': [1, 0],
            'Dependents': [0, 0],
            'tenure': [0.5, 0.1],
            'MonthlyCharges': [0.6, 0.9],
            'TotalCharges': [0.4, 0.2],
            'OnlineSecurity': [1, 0],
            'OnlineBackup': [1, 0],
            'DeviceProtection': [0, 0],
            'TechSupport': [1, 0],
            'Contract': [1, 0],
            'PaperlessBilling': [1, 1],
            'PaymentMethod': [0, 0]
        })
        
        st.dataframe(example_df, use_container_width=True)

# ============================================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================================
elif page == "📈 Model Performance":
    st.markdown('<h1 class="main-header">📈 Model Performance</h1>', unsafe_allow_html=True)
    
    try:
        # Get model info
        model_info = api_client.get_model_info()
        
        if model_info and "model_info" in model_info:
            info = model_info["model_info"]
            
            # Metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("F1 Score", f"{info.get('f1_score', 0):.2%}")
            
            with col2:
                st.metric("ROC AUC", f"{info.get('roc_auc', 0):.2%}")
            
            with col3:
                st.metric("Accuracy", f"{info.get('accuracy', 0):.2%}")
            
            with col4:
                st.metric("Run ID", info.get('run_id', 'Unknown')[:8] + "...")
            
            st.markdown("---")
            
            # Performance visualization
            st.subheader("📊 Performance Metrics")
            
            metrics_df = pd.DataFrame({
                'Metric': ['F1 Score', 'ROC AUC', 'Accuracy'],
                'Value': [info.get('f1_score', 0), info.get('roc_auc', 0), info.get('accuracy', 0)]
            })
            
            fig = px.bar(metrics_df, x='Metric', y='Value', 
                        title="Model Performance Metrics",
                        color='Value',
                        color_continuous_scale='Blues')
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Model details
            st.subheader("🔧 Model Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **Model Information:**
                - Run ID: `{info.get('run_id', 'Unknown')}`
                - Run Name: `{info.get('run_name', 'Unknown')}`
                - Algorithm: XGBoost
                - Features: 13 basic features
                """)
            
            with col2:
                st.markdown(f"""
                **Performance:**
                - F1 Score: {info.get('f1_score', 0):.4f}
                - ROC AUC: {info.get('roc_auc', 0):.4f}
                - Accuracy: {info.get('accuracy', 0):.4f}
                - Version: Production v1.0
                """)
    
    except Exception as e:
        st.error(f"Error loading model info: {e}")

# ============================================================================
# PAGE 5: ANALYTICS
# ============================================================================
elif page == "📊 Analytics":
    st.markdown('<h1 class="main-header">📊 Analytics & Insights</h1>', unsafe_allow_html=True)
    
    st.info("📊 This section would show historical predictions and trends. For now, it shows example analytics.")
    
    # Example data
    dates = pd.date_range(start='2024-01-01', end='2024-11-05', freq='D')
    np.random.seed(42)
    
    analytics_df = pd.DataFrame({
        'Date': dates,
        'Predictions': np.random.randint(50, 200, len(dates)),
        'Churn_Rate': np.random.uniform(0.2, 0.4, len(dates)),
        'High_Risk': np.random.randint(10, 50, len(dates))
    })
    
    # Time series
    st.subheader("📈 Prediction Trends Over Time")
    
    fig = px.line(analytics_df, x='Date', y='Predictions', 
                 title='Daily Predictions')
    st.plotly_chart(fig, use_container_width=True)
    
    # Churn rate
    fig = px.line(analytics_df, x='Date', y='Churn_Rate',
                 title='Churn Rate Trend')
    st.plotly_chart(fig, use_container_width=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", f"{analytics_df['Predictions'].sum():,}")
    
    with col2:
        st.metric("Avg Churn Rate", f"{analytics_df['Churn_Rate'].mean():.1%}")
    
    with col3:
        st.metric("Total High Risk", f"{analytics_df['High_Risk'].sum():,}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Customer Churn Prediction Dashboard v1.0 | Powered by MLflow + FastAPI + Streamlit</p>
</div>
""", unsafe_allow_html=True)