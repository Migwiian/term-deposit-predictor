from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import pandas as pd

# 1. load pipeline once at start-up
model = joblib.load("model.bin")

# 2. describe EXACTLY the fields the model expects
class Client(BaseModel):
    age        : int  = Field(..., ge=18, le=100)
    job        : Literal["admin.","blue-collar","technician","services","management","retired","student","unemployed","housemaid","entrepreneur","self-employed","unknown"]
    default    : Literal["yes","no","unknown"]
    housing    : Literal["yes","no","unknown"]
    loan       : Literal["yes","no","unknown"]
    marital    : Literal["married","single","divorced","unknown"]
    education  : Literal["basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown"]

class PredictionOut(BaseModel):
    subscribe_probability: float
    subscribe           : bool

app = FastAPI(title="Bank-Deposit-Predictor")

@app.post("/predict", response_model=PredictionOut)
def predict(client: Client):
    df = pd.DataFrame([client.dict()])
    proba = model.predict_proba(df.to_dict(orient="records"))[0, 1]
    return PredictionOut(
        subscribe_probability=proba,
        subscribe=proba >= 0.5   # you can tune this threshold later
    )

@app.get("/ping")
def ping():
    return "pong"

# local run:  uvicorn predict:app --reload --host 0.0.0.0 --port 8000