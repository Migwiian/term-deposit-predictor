import os
import tempfile
import importlib
import joblib
import numpy as np
from fastapi.testclient import TestClient


class _DummyModel:
    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))


def _load_predict_module():
    tmp_dir = tempfile.mkdtemp()
    model_path = os.path.join(tmp_dir, "model.bin")
    bundle = {"model": _DummyModel(), "threshold": 0.5, "metadata": {}, "metrics": {}}
    joblib.dump(bundle, model_path)
    os.environ["MODEL_PATH"] = model_path
    return importlib.import_module("predict")


predict = _load_predict_module()


def test_predict_accepts_unknown_category():
    client = TestClient(predict.app)
    payload = {
        "age": 42,
        "job": "brand-new-job",
        "marital": "married",
        "education": "tertiary",
        "default": "no",
        "balance": 1000.0,
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "day": 10,
        "month": "may",
        "campaign": 1,
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert 0.0 <= data["subscribe_probability"] <= 1.0
    assert isinstance(data["subscribe"], bool)
