from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import pandas as pd

FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
]
TARGET = "y"

AGE_MIN = 18
AGE_MAX = 100

ALLOWED_CATEGORIES: Dict[str, set[str]] = {
    "job": {
        "admin.",
        "blue-collar",
        "technician",
        "services",
        "management",
        "retired",
        "student",
        "unemployed",
        "housemaid",
        "entrepreneur",
        "self-employed",
        "unknown",
    },
    "marital": {"married", "single", "divorced", "unknown"},
    "default": {"yes", "no", "unknown"},
    "housing": {"yes", "no", "unknown"},
    "loan": {"yes", "no", "unknown"},
    "education": {
        "basic.4y",
        "basic.6y",
        "basic.9y",
        "high.school",
        "illiterate",
        "professional.course",
        "university.degree",
        "primary",
        "secondary",
        "tertiary",
        "unknown",
    },
    "contact": {"cellular", "telephone", "unknown"},
    "month": {
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "unknown",
    },
    "poutcome": {"success", "failure", "other", "unknown"},
}

NUMERIC_FEATURES = {"age", "balance", "day", "campaign", "pdays", "previous"}
CATEGORICAL_FEATURES = set(FEATURES) - NUMERIC_FEATURES


def _normalize_string(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "unknown"
    return str(value).strip().lower() or "unknown"


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    # numeric
    age = record.get("age", None)
    age = pd.to_numeric(age, errors="coerce")
    if pd.isna(age):
        age = AGE_MIN
    normalized["age"] = int(max(AGE_MIN, min(AGE_MAX, age)))

    for col in NUMERIC_FEATURES:
        if col == "age":
            continue
        val = pd.to_numeric(record.get(col, None), errors="coerce")
        if pd.isna(val):
            val = 0
        normalized[col] = float(val)

    # categoricals
    for col in CATEGORICAL_FEATURES:
        val = _normalize_string(record.get(col, None))
        if val not in ALLOWED_CATEGORIES[col]:
            val = "unknown"
        normalized[col] = val

    return normalized


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()

    # numeric
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    median_age = int(out["age"].median()) if out["age"].notna().any() else AGE_MIN
    out["age"] = out["age"].fillna(median_age).clip(AGE_MIN, AGE_MAX).astype(int)

    for col in NUMERIC_FEATURES:
        if col == "age":
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        median_val = out[col].median() if out[col].notna().any() else 0
        out[col] = out[col].fillna(median_val)

    # categoricals
    for col in CATEGORICAL_FEATURES:
        out[col] = out[col].map(_normalize_string)
        out[col] = out[col].where(out[col].isin(ALLOWED_CATEGORIES[col]), "unknown")

    # target
    out[TARGET] = out[TARGET].map(_normalize_string)
    return out
