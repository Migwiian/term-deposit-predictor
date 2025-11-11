#!/usr/bin/env python3
"""
train.py
Train a subscription predictor on UCI Bank Marketing data
and serialise the pipeline to model.bin
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

DATA_PATH   = "data/bank-full.csv"
MODEL_PATH  = "model.bin"
TEST_SIZE   = 0.2
RANDOM_STATE = 42
FEATURES = ['age','job','default','housing','loan','marital','education']
TARGET   = 'y'

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=';')

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

def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def main():
    df  = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_pipeline(X_train, y_train)
    save_model(model, MODEL_PATH)

    # quick validation
    from sklearn.metrics import roc_auc_score
    prob = model.predict_proba(X_test.to_dict(orient='records'))[:, 1]
    print("Hold-out ROC-AUC:", roc_auc_score(y_test, prob))

if __name__ == "__main__":
    main()