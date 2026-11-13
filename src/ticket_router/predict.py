"""Инференс: загрузка сохранённого pipeline и предсказание queue."""

from __future__ import annotations

from pathlib import Path

import joblib

_MODEL = None
_MODEL_PATH: Path | None = None


class NotTrainedError(RuntimeError):
    """Модель не загружена."""


def load_model(path: Path | None = None) -> None:
    """Загружает модель в память процесса (вызывается один раз при старте)."""
    global _MODEL, _MODEL_PATH
    resolved = path or Path("artifacts/model.joblib")
    if not resolved.exists():
        raise FileNotFoundError(f"Файл модели не найден: {resolved}")
    _MODEL = joblib.load(resolved)
    _MODEL_PATH = resolved


def predict_queue(subject: str, body: str) -> dict:
    """Предсказание очереди для одного обращения.

    Возвращает queue, а для вероятностных моделей — confidence.
    LinearSVC не выдаёт вероятностей, поэтому confidence отсутствует
    для непrobabilistic моделей (см. docs/architecture.md).
    """
    if _MODEL is None:
        raise NotTrainedError("Модель не загружена: вызовите load_model()")
    subject = (subject or "").strip()
    body = (body or "").strip()
    text = (subject + "\n" + body).strip()
    if not text:
        raise ValueError("Текст обращения пуст: заполните subject или body")
    prediction = _MODEL.predict([text])[0]
    result: dict = {"queue": prediction}
    if hasattr(_MODEL, "predict_proba"):
        proba = _MODEL.predict_proba([text])[0]
        result["confidence"] = round(float(proba.max()), 4)
    return result


def model_path() -> Path | None:
    return _MODEL_PATH
