from pathlib import Path

import joblib

import train


def test_train_writes_model(tmp_path):
    data_path = Path("data/bank-full.csv")
    model_path = tmp_path / "model.bin"

    bundle = train.main(data_path=data_path, model_path=model_path)

    assert model_path.exists()
    assert "model" in bundle

    loaded = joblib.load(model_path)
    assert "metadata" in loaded
    assert "threshold" in loaded
