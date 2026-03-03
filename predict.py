from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging
from features import FEATURES, ALLOWED_CATEGORIES, normalize_record, AGE_MIN, AGE_MAX
from config import load_app_config

# 1. load pipeline once at start-up
logger = logging.getLogger("term_deposit_predictor")
cfg = load_app_config()
logging.basicConfig(level=cfg.log_level, format="%(asctime)s %(levelname)s %(message)s")

MODEL_PATH = cfg.model_path

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")

bundle = joblib.load(MODEL_PATH)
if not (isinstance(bundle, dict) and "model" in bundle):
    raise RuntimeError("Invalid model bundle format. Expected dict with 'model'.")

model = bundle["model"]
threshold = float(bundle.get("threshold", 0.5))
metadata = bundle.get("metadata", {})
metrics = bundle.get("metrics", {})
logger.info("Loaded model version=%s threshold=%.4f", metadata.get("model_version"), threshold)

# 2. describe EXACTLY the fields the model expects
class Client(BaseModel):
    age        : int  = Field(..., ge=AGE_MIN, le=AGE_MAX)
    job        : str
    marital    : str
    education  : str
    default    : str
    balance    : float
    housing    : str
    loan       : str
    contact    : str
    day        : int
    month      : str
    campaign   : int
    pdays      : int
    previous   : int
    poutcome   : str

class PredictionOut(BaseModel):
    subscribe_probability: float
    subscribe           : bool

app = FastAPI(title="Bank-Deposit-Predictor")

@app.post("/predict", response_model=PredictionOut)
def predict(client: Client):
    payload = normalize_record(client.model_dump())
    df = pd.DataFrame([payload])
    proba = model.predict_proba(df.to_dict(orient="records"))[0, 1]
    return PredictionOut(
        subscribe_probability=proba,
        subscribe=proba >= threshold
    )

@app.get("/ping")
def ping():
    return "pong"

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": metadata.get("model_version"),
        "trained_at": metadata.get("trained_at"),
        "threshold": threshold,
        "metrics": metrics,
        "features": FEATURES,
        "allowed_categories": {k: sorted(v) for k, v in ALLOWED_CATEGORIES.items()},
    }

# local run:  uvicorn predict:app --reload --host 0.0.0.0 --port 8000
