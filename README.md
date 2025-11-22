# term-deposit-predictor

Live API: https://term-deposit-predictor.onrender.com/docs  
Live UI: https://term-deposit-predictor-streamlit.streamlit.app  

An end-to-end ML product that turns the UCI Bank Marketing dataset into a working micro-service:  :  
- Trained on the UCI Bank Marketing data  
- 7-line logistic model → 0.65 ROC-AUC (good enough for v1)  
- Dockerised FastAPI service you can curl from anywhere  
- Streamlit front-end for non-coders  

---

### Quick test
```bash
curl -X POST https://term-deposit-predictor.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age":42,"job":"admin.","default":"no","housing":"yes","loan":"no","marital":"married","education":"university.degree"}```
Returns:  
```json
{"subscribe_probability":0.21,"subscribe":false}```

### Retrain yourself 
- git clone https://github.com/Migwiian/term-deposit-predictor 
- cd term-deposit-predictor
- python -m venv .venv && source .venv/bin/activate
- pip install -r requirements.txt
- python train.py               # produces model.bin
- uvicorn predict:app --reload  # local API on :8000

### Stack
- Stack
- Python 3.12
- scikit-learn 1.5.0
- FastAPI 0.111.0 / Uvicorn 0.30.3
- Streamlit 1.40.0 (front-end)
- Render (API) + Streamlit Cloud (UI)