import os
from pathlib import Path
import streamlit as st
import requests
import joblib

# FAST_API URL
API_URL = os.getenv("API_URL", "https://term-deposit-predictor.onrender.com/predict")
PING_URL = API_URL.replace("/predict", "/ping")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.bin"

def load_model_info():
    try:
        bundle = joblib.load(MODEL_PATH)
        if isinstance(bundle, dict):
            metrics = bundle.get("metrics", {})
            threshold = bundle.get("threshold", None)
            return metrics, threshold
    except Exception:
        pass
    return {}, None

st.set_page_config(page_title="Term-Deposit Predictor", layout="centered")

st.title("🏦 Should we contact this client for a term deposit?")
st.markdown("Fill the pre-contact features and click **Predict**.")
st.caption("Decision uses a learned threshold from model training.")

try:
    r = requests.get(PING_URL, timeout=5)
    if r.ok:
        st.success("API is up")
    else:
        st.warning("API responded but looks unhealthy")
except Exception:
    st.error("API is not reachable")

with st.expander("Details", expanded=False):
    metrics, threshold = load_model_info()
    if threshold is not None:
        st.write(f"Saved threshold: {threshold:.4f}")
    if metrics:
        st.write("Saved metrics:")
        st.json(metrics)
    else:
        st.write("Metrics not found in model.bin")

# ---- inputs ----
col1, col2, col3 = st.columns(3)
with col1:
    age = st.slider("Age", 18, 100, 42)
    job = st.selectbox("Job", ["admin.","blue-collar","technician","services","management","retired","student","unemployed","housemaid","entrepreneur","self-employed","unknown"])
    marital = st.selectbox("Marital", ["married","single","divorced","unknown"])
    education = st.selectbox("Education", ["primary","secondary","tertiary","unknown"])
    default = st.radio("Default", ["no","yes","unknown"], horizontal=True)
with col2:
    balance = st.number_input("Balance", value=0.0, step=100.0)
    housing = st.radio("Housing loan", ["yes","no","unknown"], horizontal=True)
    loan = st.radio("Personal loan", ["yes","no","unknown"], horizontal=True)
    contact = st.selectbox("Contact type", ["cellular","telephone","unknown"])
    month = st.selectbox("Month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"])
with col3:
    day = st.slider("Day of month", 1, 31, 15)
    campaign = st.number_input("Campaign contacts", value=1, step=1, min_value=0)
    pdays = st.number_input("Days since last contact", value=-1, step=1)
    previous = st.number_input("Previous contacts", value=0, step=1, min_value=0)
    poutcome = st.selectbox("Previous outcome", ["success","failure","other","unknown"])

if st.button("Predict", type="primary"):
    payload = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": day,
        "month": month,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
    }
    try:
        res = requests.post(API_URL, json=payload, timeout=10)
        res.raise_for_status()
        data = res.json()
        prob = data["subscribe_probability"]
        will_subscribe = data["subscribe"]

        st.success(f"Probability: **{prob:.1%}**")
        if will_subscribe:
            st.balloons()
            st.info("✅ Likely to subscribe — consider calling!")
        else:
            st.warning("❌ Low chance — maybe skip this lead.")
    except Exception as e:
        st.error(f"API call failed: {e}")
