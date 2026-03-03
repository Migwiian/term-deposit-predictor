from fastapi.testclient import TestClient

import predict


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
