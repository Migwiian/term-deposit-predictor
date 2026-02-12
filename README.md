# Term Deposit Predictor Microservice

An end-to-end Machine Learning product that successfully transforms the **UCI Bank Marketing dataset** into a working microservice, deployed and accessible via a live API and front-end interface.

---

## Overview

This project focuses on predicting whether a bank client will subscribe to a term deposit.

## Data Source

- UCI Bank Marketing dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing

| Component | Description |
| :--- | :--- |
| **Model Type** | 7-line **Logistic Model** |
| **Performance (v1)** | **0.65 ROC-AUC** (Ready for initial deployment) |
| **API** | Dockerised **FastAPI** service (Supports `curl` and other HTTP requests) |
| **Front-End** | **Streamlit** UI for non-coders |

---

## Live Deployments

| Service | URL |
| :--- | :--- |
| **Live API (Docs)** | https://term-deposit-predictor.onrender.com/docs |
| **Live UI** | https://term-deposit-predictor-streamlit.streamlit.app |

---
## Local Setup (uv)

Create and sync the environment:
```bash
uv venv
uv sync
```

For the Streamlit UI dependencies:
```bash
uv sync --extra ui
```

Run training:
```bash
uv run python train.py
```

Run API:
```bash
uv run uvicorn predict:app --reload --host 0.0.0.0 --port 8000
```

Run Streamlit (local API):
```bash
API_URL=http://localhost:8000/predict uv run streamlit run app.py
```

---
### Quick Test
```bash
curl -X POST [https://term-deposit-predictor.onrender.com/predict](https://term-deposit-predictor.onrender.com/predict) \
  -H "Content-Type: application/json" \
  -d '{"age":42,"job":"admin.","default":"no","housing":"yes","loan":"no","marital":"married","education":"university.degree"}'
```

### Stack

* **Language:** Python 3.12
* **ML Library:** scikit-learn 1.5.0
* **API Framework:** FastAPI 0.111.0 / Uvicorn 0.30.3
* **Front-End:** Streamlit 1.40.0
* **Deployment:** Render (API) + Streamlit Cloud (UI)

---
## EDA Improvements (v2)

EDA is split into:

- `notebooks/01_eda.ipynb`: baseline exploration of the full raw dataset.
- `notebooks/02_eda.ipynb`: decision-driven EDA for preprocessing and validation choices.

### What improved

- Target health is explicitly measured:
  - Class balance: `88.3% no` vs `11.7% yes`
  - Majority-class baseline accuracy: `0.883`
- Unknown-value audit added by feature:
  - `education` has the highest unknown rate (`~4.1%`)
  - `job` has a smaller unknown rate (`~0.64%`)
- Age signal analysis added:
  - Older segments, especially `65+`, show stronger positive response rates
- Split diagnostics added:
  - Train/test target rates remain aligned (`~11.7%`)
  - No unseen category values in test for selected categorical features
- Duplicate interpretation corrected:
  - Full raw dataset has `0` exact duplicates across all columns
  - High duplicate counts observed earlier came from repeated profiles in a reduced feature subset, not from duplicate raw rows

### Artifacts generated

- `notebooks/artifacts/02_eda_decisions.csv`
- `notebooks/artifacts/02_eda_unknown_rates.csv`
- `notebooks/artifacts/02_eda_split_coverage.csv`
