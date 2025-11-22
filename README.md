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
### Quick Test
```bash
curl -X POST [https://term-deposit-predictor.onrender.com/predict](https://term-deposit-predictor.onrender.com/predict) \
  -H "Content-Type: application/json" \
  -d '{"age":42,"job":"admin.","default":"no","housing":"yes","loan":"no","marital":"married","education":"university.degree"}' ```

### Stack

* **Language:** Python 3.12
* **ML Library:** scikit-learn 1.5.0
* **API Framework:** FastAPI 0.111.0 / Uvicorn 0.30.3
* **Front-End:** Streamlit 1.40.0
* **Deployment:** Render (API) + Streamlit Cloud (UI)