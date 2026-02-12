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

st.title("🏦 Will this client subscribe a term deposit?")
st.markdown("Adjust the sliders / drop-downs and click **Predict**.")
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
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, 42)
    job = st.selectbox("Job", ["admin.","blue-collar","technician","services","management","retired","student","unemployed","housemaid","entrepreneur","self-employed","unknown"])
    default = st.radio("Default", ["no","yes","unknown"], horizontal=True)
    housing = st.radio("Housing loan", ["yes","no","unknown"], horizontal=True)
with col2:
    loan = st.radio("Personal loan", ["yes","no","unknown"], horizontal=True)
    marital = st.selectbox("Marital", ["married","single","divorced","unknown"])
    education = st.selectbox("Education", ["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"])

if st.button("Predict", type="primary"):
    payload = {
        "age": age,
        "job": job,
        "default": default,
        "housing": housing,
        "loan": loan,
        "marital": marital,
        "education": education
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
