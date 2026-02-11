# 🏦 Term Deposit Predictor Microservice

An end-to-end Machine Learning product that successfully transforms the **UCI Bank Marketing dataset** into a working microservice, deployed and accessible via a live API and front-end interface.

---

## ✨ Overview

This project focuses on predicting whether a bank client will subscribe to a term deposit.

| Component | Description |
| :--- | :--- |
| **Model Type** | 7-line **Logistic Model** |
| **Performance (v1)** | **0.65 ROC-AUC** (Ready for initial deployment) |
| **API** | Dockerised **FastAPI** service (Supports `curl` and other HTTP requests) |
| **Front-End** | **Streamlit** UI for non-coders |

---

## 🔗 Live Deployments

| Service | URL |
| :--- | :--- |
| **Live API (Docs)** | https://term-deposit-predictor.onrender.com/docs |
| **Live UI** | https://term-deposit-predictor-streamlit.streamlit.app |

---
## 🧰 Local Setup (uv)

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
