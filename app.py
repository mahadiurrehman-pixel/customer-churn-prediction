# app.py — Customer Churn Prediction (Matches your trained pipeline)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Tuple, Optional

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard | Customer Churn Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
def inject_custom_css() -> None:
    st.markdown("""
    <style>
        #MainMenu, header, footer {visibility: hidden;}
        .stDeployButton, div[data-testid="stToolbar"],
        div[data-testid="stDecoration"], div[data-testid="stStatusWidget"] {display: none;}

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [data-testid="stAppViewContainer"], .stApp, .main {
            background-color: #F5F7FA !important;
            color: #111827 !important;
            font-family: 'Inter', sans-serif !important;
        }

        [data-testid="stSidebar"] {
            background-color: #FAFBFC !important;
            border-right: 1px solid #E8ECF0 !important;
        }
        [data-testid="stSidebar"] * { color: #111827 !important; }

        .main .block-container {
            padding-top: 2rem; padding-bottom: 4rem; max-width: 1200px;
        }

        .app-header {
            background: #FFFFFF; border: 1px solid #E8ECF0;
            border-radius: 16px; padding: 32px 36px; margin-bottom: 28px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        .app-logo-row { display: flex; align-items: center; gap: 12px; margin-bottom: 16px; }
        .app-logo-icon {
            width: 44px; height: 44px;
            background: linear-gradient(135deg, #4F46E5, #6366F1);
            border-radius: 10px; display: flex; align-items: center;
            justify-content: center; font-size: 22px; color: white;
        }
        .app-logo-text { font-size: 22px; font-weight: 700; color: #111827; }
        .app-title { font-size: 30px; font-weight: 700; color: #111827; margin-bottom: 10px; }
        .app-description { font-size: 15px; color: #6B7280; line-height: 1.6; max-width: 720px; }

        .section-header { font-size: 18px; font-weight: 600; color: #111827; margin-bottom: 4px; }
        .section-subheader { font-size: 14px; color: #9CA3AF; margin-bottom: 20px; }

        .card {
            background: #FFFFFF; border: 1px solid #E8ECF0;
            border-radius: 12px; padding: 24px; margin-bottom: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }

        .result-card { border-radius: 16px; padding: 32px; margin: 16px 0; text-align: center; }
        .result-safe { background: #F0FDF4; border: 1px solid #BBF7D0; }
        .result-risk { background: #FFF7ED; border: 1px solid #FED7AA; }
        .result-icon { font-size: 48px; margin-bottom: 12px; }
        .result-title { font-size: 24px; font-weight: 700; margin-bottom: 8px; }
        .result-safe .result-title { color: #166534; }
        .result-risk .result-title { color: #9A3412; }
        .result-subtitle { font-size: 14px; color: #6B7280; }

        .rec-item {
            background: #F9FAFB; border: 1px solid #F3F4F6; border-radius: 8px;
            padding: 14px 18px; margin-bottom: 8px; font-size: 14px;
            color: #374151; display: flex; align-items: center; gap: 10px;
        }
        .rec-dot-green { width: 8px; height: 8px; background: #22C55E; border-radius: 50%; }
        .rec-dot-orange { width: 8px; height: 8px; background: #F97316; border-radius: 50%; }

        .prob-bar-bg { background: #F3F4F6; border-radius: 8px; height: 10px; overflow: hidden; margin: 10px 0; }
        .prob-bar-green { height: 100%; background: linear-gradient(90deg, #22C55E, #16A34A); }
        .prob-bar-orange { height: 100%; background: linear-gradient(90deg, #F97316, #EA580C); }

        .stButton > button {
            background: linear-gradient(135deg, #4F46E5, #6366F1) !important;
            color: white !important; border: none !important; border-radius: 10px !important;
            padding: 14px 32px !important; font-size: 16px !important;
            font-weight: 600 !important; width: 100% !important;
            box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25) !important;
        }

        div[data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid #E8ECF0;
            border-radius: 12px; padding: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }
        div[data-testid="stMetric"] label {
            font-size: 12px !important; font-weight: 600 !important;
            color: #6B7280 !important; text-transform: uppercase !important;
        }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
            font-size: 26px !important; font-weight: 700 !important; color: #111827 !important;
        }

        div[data-testid="stExpander"] {
            background: #FFFFFF; border: 1px solid #E8ECF0;
            border-radius: 12px; margin-bottom: 12px;
        }

        .fi-row { margin-bottom: 14px; }
        .fi-label { font-size: 13px; color: #374151; font-weight: 500; margin-bottom: 6px; }
        .fi-bar-bg { background: #F3F4F6; border-radius: 6px; height: 8px; overflow: hidden; }
        .fi-bar-fill { height: 100%; background: linear-gradient(90deg, #6366F1, #4F46E5); }
        .fi-value { font-size: 11px; color: #9CA3AF; text-align: right; margin-top: 2px; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    for name in ["churn_model.pkl", "customer_churn_xgb.pkl"]:
        if os.path.exists(name):
            return joblib.load(name)
    return None


@st.cache_resource(show_spinner=False)
def load_scaler():
    if os.path.exists("scaler.pkl"):
        return joblib.load("scaler.pkl")
    return None


# ─────────────────────────────────────────────────────────────
# Feature Order (EXACT from your training pipeline)
# ─────────────────────────────────────────────────────────────
FEATURE_ORDER: List[str] = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Paperless Billing",
    "Monthly Charges",
    "Total Charges",
    "Internet Service_Fiber optic",
    "Internet Service_No",
    "Contract_One year",
    "Contract_Two year",
    "Payment Method_Credit card (automatic)",
    "Payment Method_Electronic check",
    "Payment Method_Mailed check",
]

NUMERICAL_FEATURES: List[str] = ["Tenure Months", "Monthly Charges", "Total Charges"]


# ─────────────────────────────────────────────────────────────
# Preprocessing (Matches your notebook exactly)
# ─────────────────────────────────────────────────────────────
def preprocess_input(inputs: Dict, scaler) -> pd.DataFrame:
    """
    Preprocessing that EXACTLY matches training:
    - Gender: LabelEncoder (Female=0, Male=1)
    - Binary Yes/No → 1/0
    - Support cols: 'No internet service' → 'No' → 0
    - Multiple Lines: 'No phone service' → 'No' → 0
    - One-hot: Internet Service, Contract, Payment Method (drop_first=True)
    - StandardScaler on numerical
    """

    # ── Gender: LabelEncoder alphabetical (Female=0, Male=1) ──
    gender_val = 1 if inputs["gender"] == "Male" else 0

    # ── Binary Yes/No mapping ──
    yn = lambda v: 1 if v == "Yes" else 0

    data = {
        "Gender": gender_val,
        "Senior Citizen": yn(inputs["senior_citizen"]),
        "Partner": yn(inputs["partner"]),
        "Dependents": yn(inputs["dependents"]),
        "Tenure Months": float(inputs["tenure"]),
        "Phone Service": yn(inputs["phone_service"]),
        "Multiple Lines": yn(inputs["multiple_lines"]),  # No phone service → No → 0
        "Online Security": yn(inputs["online_security"]),
        "Online Backup": yn(inputs["online_backup"]),
        "Device Protection": yn(inputs["device_protection"]),
        "Tech Support": yn(inputs["tech_support"]),
        "Streaming TV": yn(inputs["streaming_tv"]),
        "Streaming Movies": yn(inputs["streaming_movies"]),
        "Paperless Billing": yn(inputs["paperless_billing"]),
        "Monthly Charges": float(inputs["monthly_charges"]),
        "Total Charges": float(inputs["total_charges"]),
    }

    # ── One-hot encoding (drop_first=True → drops 'DSL', 'Month-to-month', 'Bank transfer') ──
    internet = inputs["internet_service"]
    data["Internet Service_Fiber optic"] = 1 if internet == "Fiber optic" else 0
    data["Internet Service_No"] = 1 if internet == "No" else 0

    contract = inputs["contract"]
    data["Contract_One year"] = 1 if contract == "One year" else 0
    data["Contract_Two year"] = 1 if contract == "Two year" else 0

    payment = inputs["payment_method"]
    data["Payment Method_Credit card (automatic)"] = 1 if payment == "Credit card (automatic)" else 0
    data["Payment Method_Electronic check"] = 1 if payment == "Electronic check" else 0
    data["Payment Method_Mailed check"] = 1 if payment == "Mailed check" else 0

    # Build DataFrame in exact feature order
    df = pd.DataFrame([data])
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_ORDER]

    # Scale numerical features
    if scaler is not None:
        df[NUMERICAL_FEATURES] = scaler.transform(df[NUMERICAL_FEATURES])

    return df


def get_prediction(model, df: pd.DataFrame) -> Tuple[int, float]:
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])
    return prediction, probability


def get_risk_level(probability: float) -> str:
    if probability < 0.3: return "Low"
    elif probability < 0.6: return "Medium"
    elif probability < 0.8: return "High"
    else: return "Critical"


def get_confidence(probability: float) -> float:
    return abs(probability - 0.5) * 2


# ─────────────────────────────────────────────────────────────
# UI Components
# ─────────────────────────────────────────────────────────────
def render_header() -> None:
    html = (
        '<div class="app-header">'
        '<div class="app-logo-row">'
        '<div class="app-logo-icon">🛡️</div>'
        '<div class="app-logo-text">ChurnGuard</div>'
        '</div>'
        '<div class="app-title">Customer Churn Prediction</div>'
        '<div class="app-description">'
        'Leverage machine learning to identify at-risk customers before they leave. '
        'Powered by XGBoost trained on the Telco Customer Churn dataset with 81%+ accuracy.'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🛡️ ChurnGuard")
        st.caption("Enterprise ML Platform")
        st.markdown("---")

        st.markdown("#### 📋 Project Overview")
        st.write(
            "Predicts customer churn for telecom companies using "
            "advanced gradient boosting. Trained on 7,043 real customer records."
        )
        st.markdown("---")

        st.markdown("#### 🧠 Model Information")
        st.write("**Algorithm:** XGBoost Classifier")
        st.write("**Tuning:** RandomizedSearchCV")
        st.write("**Scoring:** ROC-AUC")
        st.write("**CV Folds:** 5")
        st.write("**Features:** 23")
        st.markdown("---")

        st.markdown("#### 📊 Dataset Information")
        st.write("**Source:** Telco Customer Churn (IBM)")
        st.write("**Records:** 7,043")
        st.write("**Target:** Churn Value (Binary)")
        st.write("**Churn Rate:** ~26.5%")
        st.markdown("---")

        st.markdown("#### ⚙️ Technologies")
        st.write("Python · Streamlit · XGBoost · Scikit-learn · Pandas · NumPy · Joblib")
        st.markdown("---")

        st.markdown("#### 👤 Developer")
        st.write("Portfolio project — end-to-end ML pipeline with production UI.")


def render_dashboard_metrics() -> None:
    st.markdown('<div class="section-header">Model Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Key metrics from model training and evaluation</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Model Accuracy", "81.4%")
    with c2: st.metric("ROC-AUC Score", "0.856")
    with c3: st.metric("Dataset Size", "7,043")
    with c4: st.metric("Model Type", "XGBoost")


def render_feature_importance() -> None:
    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Top features driving predictions</div>', unsafe_allow_html=True)

    model = load_model()
    importance_data = {
        "Tenure Months": 0.182, "Monthly Charges": 0.148, "Total Charges": 0.137,
        "Contract_Two year": 0.098, "Internet Service_Fiber optic": 0.087,
        "Payment Method_Electronic check": 0.072, "Online Security": 0.061,
        "Tech Support": 0.054, "Contract_One year": 0.048, "Paperless Billing": 0.038,
    }

    if model is not None:
        try:
            importances = model.feature_importances_
            imp_dict = dict(zip(FEATURE_ORDER, importances))
            imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:10])
            if sum(imp_dict.values()) > 0:
                importance_data = imp_dict
        except Exception:
            pass

    max_val = max(importance_data.values())
    bars = ""
    for feature, value in importance_data.items():
        width_pct = (value / max_val) * 100
        bars += (
            f'<div class="fi-row">'
            f'<div class="fi-label">{feature}</div>'
            f'<div class="fi-bar-bg"><div class="fi-bar-fill" style="width:{width_pct:.1f}%;"></div></div>'
            f'<div class="fi-value">{value:.3f}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="card">{bars}</div>', unsafe_allow_html=True)


def render_prediction_result(prediction: int, probability: float) -> None:
    risk_level = get_risk_level(probability)
    confidence_pct = get_confidence(probability) * 100
    prob_pct = probability * 100

    if prediction == 0:
        html = (
            '<div class="result-card result-safe">'
            '<div class="result-icon">✅</div>'
            '<div class="result-title">Customer Likely To Stay</div>'
            '<div class="result-subtitle">Strong retention indicators detected</div>'
            '</div>'
        )
    else:
        html = (
            '<div class="result-card result-risk">'
            '<div class="result-icon">⚠️</div>'
            '<div class="result-title">High Risk Customer</div>'
            '<div class="result-subtitle">Immediate retention action recommended</div>'
            '</div>'
        )
    st.markdown(html, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    bar_class = "prob-bar-orange" if prediction == 1 else "prob-bar-green"

    with c1:
        st.markdown(
            f'<div class="card">'
            f'<div style="font-size:12px;color:#6B7280;font-weight:600;text-transform:uppercase;">Churn Probability</div>'
            f'<div style="font-size:28px;font-weight:700;color:#111827;margin-top:6px;">{prob_pct:.1f}%</div>'
            f'<div class="prob-bar-bg"><div class="{bar_class}" style="width:{prob_pct:.1f}%;"></div></div>'
            f'<div style="font-size:12px;color:#9CA3AF;">Likelihood of leaving</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f'<div class="card">'
            f'<div style="font-size:12px;color:#6B7280;font-weight:600;text-transform:uppercase;">Confidence</div>'
            f'<div style="font-size:28px;font-weight:700;color:#111827;margin-top:6px;">{confidence_pct:.1f}%</div>'
            f'<div class="prob-bar-bg"><div class="prob-bar-green" style="width:{confidence_pct:.1f}%;"></div></div>'
            f'<div style="font-size:12px;color:#9CA3AF;">Model certainty</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with c3:
        risk_color = {"Low":"#22C55E","Medium":"#EAB308","High":"#F97316","Critical":"#EF4444"}[risk_level]
        st.markdown(
            f'<div class="card">'
            f'<div style="font-size:12px;color:#6B7280;font-weight:600;text-transform:uppercase;">Risk Level</div>'
            f'<div style="font-size:28px;font-weight:700;color:{risk_color};margin-top:6px;">{risk_level}</div>'
            f'<div style="margin-top:10px;"><span style="display:inline-block;width:10px;height:10px;background:{risk_color};border-radius:50%;"></span></div>'
            f'<div style="font-size:12px;color:#9CA3AF;margin-top:4px;">Overall risk</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown('<div class="section-header">Business Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Actionable steps based on the prediction</div>', unsafe_allow_html=True)

    if prediction == 1:
        dot = "rec-dot-orange"
        recs = [
            "Offer a personalized discount or loyalty reward",
            "Assign a dedicated retention specialist for outreach",
            "Review and upgrade the customer's current plan",
            "Schedule a satisfaction survey to identify pain points",
            "Offer contract extension with improved terms",
        ]
    else:
        dot = "rec-dot-green"
        recs = [
            "Customer is stable — continue engagement strategy",
            "Consider upselling premium services",
            "Maintain regular check-ins",
            "Include in loyalty & referral programs",
        ]

    recs_html = "".join([f'<div class="rec-item"><div class="{dot}"></div>{r}</div>' for r in recs])
    st.markdown(f'<div class="card">{recs_html}</div>', unsafe_allow_html=True)


def render_information_section() -> None:
    st.markdown("---")
    st.markdown('<div class="section-header">Knowledge Base</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Learn more about the model</div>', unsafe_allow_html=True)

    with st.expander("📖  What is Customer Churn?"):
        st.write(
            "Customer churn is when customers stop doing business with a company. "
            "In telecom, retaining customers is 5-7x cheaper than acquiring new ones. "
            "Predicting churn enables proactive retention."
        )

    with st.expander("🧠  How Does the Model Work?"):
        st.write(
            "This XGBoost classifier was trained using RandomizedSearchCV with 5-fold "
            "cross-validation on ROC-AUC scoring. Feature engineering included binary "
            "encoding, one-hot encoding (drop_first), and StandardScaler on numerical features."
        )

    with st.expander("⚠️  Model Limitations"):
        st.write(
            "Trained on historical Telco data — may not generalize to other industries. "
            "Doesn't capture competitor pricing or economic factors. Use with domain expertise."
        )

    with st.expander("💼  Business Use Cases"):
        st.write(
            "Proactive retention, customer segmentation, revenue protection, "
            "product improvement, and executive reporting."
        )


def render_input_form() -> Optional[Dict]:
    st.markdown('<div class="section-header">Customer Data Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subheader">Fill in customer details</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("##### 👤 Customer Information")
        c1, c2, c3, c4 = st.columns(4)
        with c1: gender = st.selectbox("Gender", ["Male", "Female"])
        with c2: senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        with c3: partner = st.selectbox("Partner", ["No", "Yes"])
        with c4: dependents = st.selectbox("Dependents", ["No", "Yes"])

        st.markdown("##### 📡 Service Information")
        s1, s2, s3 = st.columns(3)
        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        with s2:
            online_security = st.selectbox("Online Security", ["No", "Yes"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        with s3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

        st.markdown("##### 💳 Account Information")
        a1, a2 = st.columns(2)
        with a1:
            tenure = st.number_input("Tenure (Months)", 0, 72, 12, 1)
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 0.5)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0, 10.0)
        with a2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Bank transfer (automatic)", "Credit card (automatic)",
                "Electronic check", "Mailed check"
            ])

        st.markdown("")
        submitted = st.form_submit_button("🔍  Predict Churn", use_container_width=True)

        if submitted:
            return {
                "gender": gender, "senior_citizen": senior_citizen, "partner": partner,
                "dependents": dependents, "phone_service": phone_service,
                "multiple_lines": multiple_lines, "internet_service": internet_service,
                "online_security": online_security, "online_backup": online_backup,
                "device_protection": device_protection, "tech_support": tech_support,
                "streaming_tv": streaming_tv, "streaming_movies": streaming_movies,
                "tenure": tenure, "monthly_charges": monthly_charges,
                "total_charges": total_charges, "contract": contract,
                "paperless_billing": paperless_billing, "payment_method": payment_method,
            }
    return None


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main() -> None:
    inject_custom_css()
    model = load_model()
    scaler = load_scaler()

    render_sidebar()
    render_header()

    if model is None:
        st.error("⚠️ Model file not found. Place `churn_model.pkl` (or `customer_churn_xgb.pkl`) in the app folder.")
        st.stop()

    if scaler is None:
        st.warning("⚠️ Scaler file not found. Predictions may be inaccurate. Save `scaler.pkl` from training.")

    render_dashboard_metrics()
    st.markdown("---")

    form_col, viz_col = st.columns([3, 2], gap="large")
    with form_col:
        inputs = render_input_form()
    with viz_col:
        render_feature_importance()

    if inputs is not None:
        st.markdown("---")
        with st.spinner("Analyzing customer data..."):
            try:
                df = preprocess_input(inputs, scaler)
                prediction, probability = get_prediction(model, df)
                render_prediction_result(prediction, probability)
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                st.exception(e)

    render_information_section()


if __name__ == "__main__":
    main()