"""REST API для маршрутизации тикетов (FastAPI).

Эндпоинты:
- POST /predict  — классификация обращения;
- GET  /health   — живость процесса (всегда 200, если процесс отвечает);
- GET  /ready    — готовность к работе: 503, пока модель не загружена;
- GET  /metrics  — технические счётчики и распределение предсказаний.
"""

from __future__ import annotations

import time
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .predict import load_model, model_path, predict_queue
from .webui import HTML_PAGE

_stats: Counter = Counter()  # requests_total, errors_4xx, errors_5xx, predictions_total
_queue_dist: Counter = Counter()
_started_at = time.time()


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Загружаем артефакт один раз при старте процесса.
    try:
        load_model()
    except FileNotFoundError:
        # Приложение поднимется, но /ready будет возвращать 503 —
        # это отделено от /health, которую использует проверка живости.
        pass
    yield


app = FastAPI(title="ticket-router API", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Учебные технические метрики: запросы и ошибки по классам статусов."""
    _stats["requests_total"] += 1
    response = await call_next(request)
    if 400 <= response.status_code < 500:
        _stats["errors_4xx"] += 1
    elif response.status_code >= 500:
        _stats["errors_5xx"] += 1
    return response


class TicketIn(BaseModel):
    subject: str = Field(default="", description="Тема обращения")
    body: str = Field(description="Текст обращения (обязательное поле)")


class PredictOut(BaseModel):
    queue: str
    confidence: float | None = None


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    """Веб-демо с имитацией чата (подход заимствован у ai_ticket_classifier).

    Использует тот же POST /predict, что и программные клиенты.
    """
    return HTML_PAGE


@app.get("/health")
def health() -> dict:
    """Живость процесса. Готовность модели — в /ready."""
    return {
        "status": "ok",
        "model_loaded": model_path() is not None,
        "version": __version__,
        "uptime_seconds": round(time.time() - _started_at, 1),
    }


@app.get("/ready")
def ready() -> dict:
    """Readiness-проверка: 503, пока артефакт модели не загружен."""
    if model_path() is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    return {"status": "ready"}


@app.get("/metrics")
def metrics() -> dict:
    """Учебные технические и модельные метрики (см. docs/deployment_monitoring.md)."""
    total = max(_stats["requests_total"], 1)
    return {
        "requests_total": _stats["requests_total"],
        "errors_4xx": _stats["errors_4xx"],
        "errors_5xx": _stats["errors_5xx"],
        "error_rate": round((_stats["errors_4xx"] + _stats["errors_5xx"]) / total, 4),
        "predictions_total": _stats["predictions_total"],
        "predicted_queues": dict(_queue_dist),
    }


@app.post("/predict", response_model=PredictOut)
def predict(ticket: TicketIn) -> PredictOut:
    if not ticket.body.strip() and not ticket.subject.strip():
        raise HTTPException(status_code=422, detail="subject и body пусты")
    if model_path() is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    try:
        result = predict_queue(ticket.subject, ticket.body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _stats["predictions_total"] += 1
    _queue_dist[result["queue"]] += 1
    return PredictOut(queue=result["queue"], confidence=result.get("confidence"))
