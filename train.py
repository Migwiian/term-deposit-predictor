#!/usr/bin/env python3
"""
train.py
Train a subscription predictor on UCI Bank Marketing data
and serialise the pipeline to model.bin
"""

from pathlib import Path
from datetime import datetime, timezone
import os
import pandas as pd
import joblib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score, precision_recall_curve
from features import FEATURES, TARGET, normalize_dataframe
from config import load_train_config

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "bank-full.csv",
    BASE_DIR / "data" / "bank.csv",
    BASE_DIR / "data" / "bank-clean.csv",
]
MODEL_PATH = BASE_DIR / "model.bin"
TEST_SIZE   = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42

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

def split_train_val_test(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # val size as fraction of remaining train
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_pipeline():
    return make_pipeline(
        DictVectorizer(),
        LogisticRegression(max_iter=1000, class_weight='balanced')
    )

def train_pipeline(X_train, y_train):
    pipe = build_pipeline()
    pipe.fit(X_train.to_dict(orient='records'), y_train)
    return pipe

def pick_threshold_f1(y_true, prob):
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = (2 * precision * recall) / (precision + recall + 1e-12)
    best_idx = int(f1.argmax())
    return float(thresholds[best_idx]), float(f1[best_idx]), float(precision[best_idx]), float(recall[best_idx])

def pick_threshold_fbeta(y_true, prob, beta: float):
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    precision = precision[:-1]
    recall = recall[:-1]
    beta2 = beta * beta
    fbeta = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-12)
    best_idx = int(fbeta.argmax())
    return float(thresholds[best_idx]), float(fbeta[best_idx]), float(precision[best_idx]), float(recall[best_idx])

def save_model(bundle, path: Path):
    joblib.dump(bundle, path)
    print(f"Model saved to {path}")

def main(data_path: Path | None = None, model_path: Path | None = None):
    cfg = load_train_config()
    data_path = data_path or cfg.data_path or resolve_data_path()
    model_path = model_path or cfg.model_path
    print(f"Using dataset: {data_path}")
    df  = load_data(data_path)
    df = normalize_dataframe(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(df)
    model = train_pipeline(X_train, y_train)

    # choose threshold on validation set
    val_prob = model.predict_proba(X_val.to_dict(orient='records'))[:, 1]
    y_val_bin = (y_val == "yes").astype(int)
    if cfg.threshold_strategy == "f1":
        threshold, fscore, precision, recall = pick_threshold_f1(y_val_bin, val_prob)
        fscore_name = "f1"
    elif cfg.threshold_strategy == "fbeta":
        threshold, fscore, precision, recall = pick_threshold_fbeta(y_val_bin, val_prob, cfg.threshold_beta)
        fscore_name = f"fbeta_{cfg.threshold_beta}"
    else:
        raise ValueError(f"Unknown THRESHOLD_STRATEGY: {cfg.threshold_strategy}")
    val_roc_auc = roc_auc_score(y_val_bin, val_prob)

    # evaluate on test set using chosen threshold
    test_prob = model.predict_proba(X_test.to_dict(orient='records'))[:, 1]
    y_test_bin = (y_test == "yes").astype(int)
    test_roc_auc = roc_auc_score(y_test_bin, test_prob)
    test_pred = (test_prob >= threshold).astype(int)
    test_precision = (test_pred & y_test_bin).sum() / max(test_pred.sum(), 1)
    test_recall = (test_pred & y_test_bin).sum() / max(y_test_bin.sum(), 1)
    test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall + 1e-12)

    print("Validation ROC-AUC:", val_roc_auc)
    print("Chosen threshold:", round(threshold, 4))
    print(f"Validation {fscore_name}:", round(fscore, 4))
    print("Validation precision:", round(precision, 4))
    print("Validation recall:", round(recall, 4))
    print("Test ROC-AUC:", round(test_roc_auc, 4))
    print("Test F1:", round(test_f1, 4))
    print("Test precision:", round(test_precision, 4))
    print("Test recall:", round(test_recall, 4))

    model_version = cfg.model_version or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    bundle = {
        "model": model,
        "threshold": threshold,
        "metrics": {
            "val_roc_auc": float(val_roc_auc),
            "val_f1": fscore if fscore_name == "f1" else None,
            "val_precision": precision,
            "val_recall": recall,
            "test_roc_auc": float(test_roc_auc),
            "test_f1": float(test_f1),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "threshold_score_name": fscore_name,
            "threshold_score_value": float(fscore),
        },
        "metadata": {
            "model_version": model_version,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "data_path": str(data_path),
            "features": FEATURES,
            "val_size": VAL_SIZE,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "sklearn_version": sklearn.__version__,
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "threshold_strategy": cfg.threshold_strategy,
            "threshold_beta": cfg.threshold_beta,
        },
    }
    save_model(bundle, model_path)
    return bundle

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the term deposit predictor.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--model-path", type=str, default=None, help="Output path for model.bin")
    args = parser.parse_args()

    data_path = Path(args.data_path) if args.data_path else None
    model_path = Path(args.model_path) if args.model_path else None
    main(data_path=data_path, model_path=model_path)
