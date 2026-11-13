"""Инференс: загрузка сохранённого pipeline и предсказание queue."""

from __future__ import annotations

import os
import re
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

_MODEL = None
_MODEL_PATH: Path | None = None


class NotTrainedError(RuntimeError):
    """Модель не загружена."""


class InvalidTicketError(ValueError):
    """Некорректный пользовательский текст."""


class UnsupportedTextError(InvalidTicketError):
    """Текст вне поддерживаемого моделью входа; очередь не назначается."""


def load_model(path: Path | None = None) -> None:
    """Загружает модель в память процесса (вызывается один раз при старте)."""
    global _MODEL, _MODEL_PATH
    # При неудачной повторной загрузке нельзя оставлять старую readiness.
    _MODEL = None
    _MODEL_PATH = None
    resolved = Path(path or os.environ.get("TICKET_ROUTER_MODEL_PATH", "artifacts/model.joblib"))
    if not resolved.exists():
        raise FileNotFoundError(f"Файл модели не найден: {resolved}")
    model = joblib.load(resolved)
    pipeline = getattr(model, "best_estimator_", model)
    if not isinstance(pipeline, Pipeline) or len(pipeline.steps) < 2:
        raise ValueError("Ожидается обученный pipeline: векторизация + классификатор")
    check_is_fitted(pipeline)
    _MODEL = model
    _MODEL_PATH = resolved.resolve()


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
        raise InvalidTicketError("Текст обращения пуст: заполните subject или body")
    # Это проверка кириллицы, а не определитель языка. Остальные языки
    # не гарантированно распознаются: область применения модели — EN/DE.
    if re.search(r"[\u0400-\u052f]", text):
        raise UnsupportedTextError(
            "Модель обучена только на английском и немецком (EN/DE). "
            "Кириллица не поддерживается; очередь не назначена. "
            "Передайте обращение оператору или введите текст на EN/DE."
        )
    pipeline = getattr(_MODEL, "best_estimator_", _MODEL)
    features = pipeline[:-1].transform([text])
    if features.nnz == 0:
        raise UnsupportedTextError(
            "В тексте нет известных модели признаков; очередь не назначена. "
            "Опишите проблему на английском или немецком либо передайте её оператору."
        )
    # Преобразуем текст один раз, тем же pipeline, что использован в обучении.
    classifier = pipeline.steps[-1][1]
    prediction = classifier.predict(features)[0]
    result: dict = {"queue": prediction}
    if hasattr(classifier, "predict_proba"):
        proba = classifier.predict_proba(features)[0]
        result["confidence"] = round(float(proba.max()), 4)
    return result


def model_path() -> Path | None:
    return _MODEL_PATH
