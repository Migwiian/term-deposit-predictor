#!/usr/bin/env python3
"""
train.py
Train a subscription predictor on UCI Bank Marketing data
and serialise the pipeline to model.bin
"""

from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, precision_recall_curve

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "bank-clean.csv",
    BASE_DIR / "data" / "bank-full.csv",
    BASE_DIR / "data" / "bank.csv",
]
MODEL_PATH = BASE_DIR / "model.bin"
TEST_SIZE   = 0.2
RANDOM_STATE = 42
FEATURES = ['age','job','default','housing','loan','marital','education']
TARGET   = 'y'

def resolve_data_path() -> Path:
    for path in DATA_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No dataset found. Tried:\n" + "\n".join(str(p) for p in DATA_CANDIDATES)
    )

def load_data(path: Path) -> pd.DataFrame:
    # Use separator auto-detection so both raw UCI files (;) and cleaned files (,) are supported.
    return pd.read_csv(path, sep=None, engine="python")

def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(X, y,
                            test_size=TEST_SIZE,
                            random_state=RANDOM_STATE,
                            stratify=y)

def build_pipeline():
    return make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(max_iter=1000, class_weight='balanced')
    )

def train_pipeline(X_train, y_train):
    pipe = build_pipeline()
    pipe.fit(X_train.to_dict(orient='records'), y_train)
    return pipe

def pick_threshold(y_true, prob):
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = int(f1.argmax())
    return float(thresholds[best_idx]), float(f1[best_idx]), float(precision[best_idx]), float(recall[best_idx])

def save_model(bundle, path: Path):
    joblib.dump(bundle, path)
    print(f"Model saved to {path}")

def main():
    data_path = resolve_data_path()
    print(f"Using dataset: {data_path}")
    df  = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_pipeline(X_train, y_train)

    # quick validation and threshold selection
    prob = model.predict_proba(X_test.to_dict(orient='records'))[:, 1]
    y_test_bin = (y_test == "yes").astype(int)
    threshold, f1, precision, recall = pick_threshold(y_test_bin, prob)
    roc_auc = roc_auc_score(y_test_bin, prob)
    print("Hold-out ROC-AUC:", roc_auc)
    print("Chosen threshold:", round(threshold, 4))
    print("Hold-out F1:", round(f1, 4))
    print("Hold-out precision:", round(precision, 4))
    print("Hold-out recall:", round(recall, 4))

    bundle = {
        "model": model,
        "threshold": threshold,
        "metrics": {
            "roc_auc": float(roc_auc),
            "f1": f1,
            "precision": precision,
            "recall": recall,
        },
    }
    save_model(bundle, MODEL_PATH)

if __name__ == "__main__":
    main()
