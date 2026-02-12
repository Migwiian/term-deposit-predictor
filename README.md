# Term Deposit Predictor

This project predicts whether a bank client will subscribe to a term deposit. It includes data prep, model training, an API for predictions, and a Streamlit UI. The goal is to show a clean junior level MLE workflow end to end.

## Live links

- API docs: https://term-deposit-predictor.onrender.com/docs
- Streamlit app: https://term-deposit-predictor-streamlit.streamlit.app

## Quick summary for recruiters

- Built a full pipeline from raw data to a deployed API and UI
- Used a simple baseline model and saved a decision threshold from hold out data
- Added metrics and artifacts for review and reproducibility

## Problem statement

A bank wants to predict who will subscribe to a term deposit. The model helps decide who to contact and who to skip, based on the predicted probability of subscription.

## Data

Source: UCI Bank Marketing dataset
https://archive.ics.uci.edu/dataset/222/bank+marketing

Target: `y` where `yes` means the client subscribed.

Features used:

- age
- job
- default
- housing
- loan
- marital
- education

## Approach

1. Clean and standardize the selected fields
2. Train a baseline logistic regression model
3. Use a hold out set to pick a decision threshold based on F1 for the positive class
4. Save the model, threshold, and metrics to a single file
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


## Model metrics from latest run

These metrics are from the most recent local training run.

- ROC AUC: 0.6643
- F1: 0.2960
- Precision: 0.2103
- Recall: 0.5000
- Decision threshold: 0.5601

## Project structure

- `train.py` trains the model and writes `model.bin`
- `predict.py` loads `model.bin` and serves the API
- `app.py` is the Streamlit UI
- `notebooks/` contains EDA and model comparison
- `data/` contains raw and cleaned data
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
  -d '{"age":42,"job":"admin.","default":"no","housing":"yes","loan":"no","marital":"married","education":"university.degree"}'
```

## Run the Streamlit UI

```bash
API_URL=http://localhost:8000/predict uv run streamlit run app.py
```

The UI includes a health check and a Details panel with the saved threshold and metrics.

## What to look at

- EDA notebooks in `notebooks/`
- Cleaned dataset at `data/bank-clean.csv`
- Model artifact at `model.bin`
- API and UI for end to end flow

## Notes for reviewers

- The data split is stratified by the target
- The model uses class weights for imbalance
- Metrics are saved in `model.bin` for quick inspection
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
