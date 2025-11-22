# bank-deposit-api 🏦

Live API (when deployed): `https://your-app.fly.dev/docs`

&gt; **Work-in-progress** – building an end-to-end ML micro-service on the UCI Bank Marketing dataset.  
&gt; Goal: train a subscription predictor, wrap it in FastAPI, containerise with Docker, deploy to the cloud.

---

### What we’re doing (high-level)
1. Exploratory analysis & feature selection (notebook)
2. Train + validate model (scikit-learn pipeline)
3. Serialise pipeline → `model.bin`
4. FastAPI service `/predict` with Pydantic validation
5. Docker image + CI → public URL
6. (Optional) tiny Streamlit front-end

---

### Project tree (so far)
bank-deposit-api/
├── data/
│   └── bank-full.csv          ← 41 k rows, 20 inputs + target
├── notebooks/
│   └── 01_eda.ipynb           ← blank, your playground
├── src/                       ← scripts will land here
├── README.md                  ← this file
└── model.bin                  ← will appear after training


### Quick start (Codespaces or local)
```bash
# 1. clone
git clone https://github.com/YOUR-USER/bank-deposit-api.git
cd bank-deposit-api

# 2. create env (optional but nice)
python -m venv .venv && source .venv/bin/activate

# 3. install core deps
pip install pandas scikit-learn jupyter seaborn joblib

# 4. open notebook
jupyter notebook notebooks/01_eda.ipynb
