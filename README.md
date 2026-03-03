# Term Deposit Predictor

This project predicts whether a bank client will subscribe to a term deposit. It includes data prep, model training, an API for predictions, and a Streamlit UI.

## Live links

- API docs: https://term-deposit-predictor.onrender.com/docs
- Streamlit app: https://term-deposit-predictor-streamlit.streamlit.app

## Quick summary

- Full pipeline from raw data to a deployed API and UI
- Baseline model with a learned decision threshold
- Metrics and artifacts saved for review and reproducibility

## Problem statement

A bank wants to decide who to contact before a campaign. The model estimates the probability of subscription and supports a contact vs skip decision.

## Data

Source: UCI Bank Marketing dataset
https://archive.ics.uci.edu/dataset/222/bank+marketing

Target: `y` where `yes` means the client subscribed.

Pre-contact features used (from `bank-full.csv`):

- age
- job
- marital
- education
- default
- balance
- housing
- loan
- contact
- day
- month
- campaign
- pdays
- previous
- poutcome

Excluded feature:
- duration (post-contact; not available at decision time)

## Approach

1. Clean and standardize the selected pre-contact fields
2. Train a baseline logistic regression model
3. Use a validation set to pick a decision threshold (F1 by default, F-beta optional)
4. Evaluate once on a test set and save metrics
5. Serve predictions through a FastAPI service
6. Provide a Streamlit UI that calls the API

## System flow

```text
raw data
  -> data prep
  -> train.py
  -> model.bin (model + threshold + metrics)
  -> predict.py (FastAPI)
  -> Streamlit UI
```


## Model metrics

Metrics are saved inside `model.bin` and exposed at `/health`. Run `uv run python train.py` to update them.

## Project structure

- `train.py` trains the model and writes `model.bin`
- `predict.py` loads `model.bin` and serves the API
- `app.py` is the Streamlit UI
- `notebooks/` contains EDA and model comparison
- `data/` contains raw data
- `model.bin` stores the model, threshold, and metrics

## Local setup

Create and sync the environment:

```bash
uv venv
uv sync
```

For Streamlit UI dependencies:

```bash
uv sync --extra ui
```

## Train the model

```bash
uv run python train.py
```

This writes `model.bin` in the project root.

## Run the API

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 8000
```

Test with curl:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":42,"job":"admin.","marital":"married","education":"tertiary","default":"no","balance":1200,"housing":"yes","loan":"no","contact":"cellular","day":15,"month":"may","campaign":1,"pdays":-1,"previous":0,"poutcome":"unknown"}'
```

Health check:

```bash
curl http://localhost:8000/health
```

## Configuration (env vars)

- `MODEL_PATH` (train + API): override path to `model.bin`
- `DATA_PATH` (train): override dataset CSV path
- `MODEL_VERSION` (train): override model version string
- `LOG_LEVEL` (API): set log level (default: `INFO`)
- `THRESHOLD_STRATEGY` (train): `f1` or `fbeta` (recall-leaning)
- `THRESHOLD_BETA` (train): beta value when `THRESHOLD_STRATEGY=fbeta` (default: `2.0`)

## Model behavior and tradeoffs

- The model supports a pre-contact decision: call vs skip.
- To maximize subscribers, use a recall-leaning threshold.
- Example: `THRESHOLD_STRATEGY=fbeta THRESHOLD_BETA=2.0 uv run python train.py`
- Higher recall means more false positives.

## Run the Streamlit UI

```bash
API_URL=http://localhost:8000/predict uv run streamlit run app.py
```

The UI includes a health check and a Details panel with the saved threshold and metrics.

## What to look at

- EDA notebooks in `notebooks/`
- Raw dataset at `data/bank-full.csv`
- Model artifact at `model.bin`
- API and UI for end to end flow

## Notes

- The data split is stratified by the target
- The model uses class weights for imbalance
- Metrics (validation + test) are saved in `model.bin` for quick inspection
- The Streamlit app calls the API and does not load the model directly


## Docker build note

The Docker image trains the model during the build, using the data in `data/`. This avoids committing `model.bin` to git and keeps the Render build simple.

## Tools and versions

- Python 3.12
- scikit learn 1.5.0
- FastAPI 0.111.0
- Streamlit 1.40.0
- Render and Streamlit Cloud for deployment

## EDA artifacts

- `notebooks/01_eda.ipynb`
- `notebooks/02_eda.ipynb`
- `notebooks/03_prep_and_model_compare.ipynb`
- `notebooks/artifacts/02_eda_decisions.csv`
- `notebooks/artifacts/02_eda_unknown_rates.csv`
- `notebooks/artifacts/02_eda_split_coverage.csv`
- `notebooks/artifacts/03_model_compare_cv.csv`

## Production-grade additions

- Input normalization with unknown-category fallback
- Model metadata/versioning stored in `model.bin`
- `/health` endpoint exposing model info
- Basic tests for training and prediction
