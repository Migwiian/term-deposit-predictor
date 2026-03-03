from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _getenv(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None:
        return default
    return value


@dataclass(frozen=True)
class AppConfig:
    model_path: Path
    log_level: str


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path | None
    model_path: Path
    model_version: str | None
    threshold_strategy: str
    threshold_beta: float


def load_app_config() -> AppConfig:
    model_path = Path(_getenv("MODEL_PATH", str(BASE_DIR / "model.bin")))
    log_level = _getenv("LOG_LEVEL", "INFO") or "INFO"
    return AppConfig(model_path=model_path, log_level=log_level)


def load_train_config() -> TrainConfig:
    data_path_raw = _getenv("DATA_PATH")
    data_path = Path(data_path_raw) if data_path_raw else None
    model_path = Path(_getenv("MODEL_PATH", str(BASE_DIR / "model.bin")))
    model_version = _getenv("MODEL_VERSION")
    threshold_strategy = (_getenv("THRESHOLD_STRATEGY", "f1") or "f1").lower()
    threshold_beta = float(_getenv("THRESHOLD_BETA", "2.0") or "2.0")
    return TrainConfig(
        data_path=data_path,
        model_path=model_path,
        model_version=model_version,
        threshold_strategy=threshold_strategy,
        threshold_beta=threshold_beta,
    )
