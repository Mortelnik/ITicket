"""REST API для маршрутизации тикетов (FastAPI).

Эндпоинты:
- POST /predict  — классификация обращения;
- GET  /health   — живость процесса (всегда 200, если процесс отвечает);
- GET  /ready    — готовность к работе: 503, пока модель не загружена;
- GET  /metrics  — технические счётчики и распределение предсказаний.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import __version__
from .predict import InvalidTicketError, UnsupportedTextError, load_model, model_path, predict_queue
from .security import RequestGuard, is_secret_path
from .webui import HTML_PAGE

_stats: Counter = Counter()  # requests_total, errors_4xx, errors_5xx, predictions_total
_queue_dist: Counter = Counter()
_started_at = time.time()
_latencies: deque[float] = deque(maxlen=1000)
_metrics_lock = threading.Lock()
_request_guard = RequestGuard.from_env()


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
    started = time.perf_counter()
    with _metrics_lock:
        _stats["requests_total"] += 1
    try:
        if is_secret_path(request.url.path):
            with _metrics_lock:
                _stats["secret_requests_total"] += 1
            response = Response(status_code=444)
        else:
            # Заголовки Forwarded/X-Forwarded-For здесь не читаются: адрес
            # клиента устанавливает ASGI-сервер с учётом доверенных proxy.
            client = request.client.host if request.client else "unknown"
            retry_after = _request_guard.retry_after(client)
            if retry_after:
                with _metrics_lock:
                    _stats["rate_limited_requests_total"] += 1
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Слишком много запросов. Повторите позже."},
                    headers={"Retry-After": str(retry_after)},
                )
            else:
                response = await call_next(request)
    except Exception:
        with _metrics_lock:
            _stats["errors_5xx"] += 1
        raise
    else:
        with _metrics_lock:
            if 400 <= response.status_code < 500:
                _stats["errors_4xx"] += 1
            elif response.status_code >= 500:
                _stats["errors_5xx"] += 1
            if request.url.path == "/predict" and response.status_code == 422:
                _stats["rejected_inputs_total"] += 1
    finally:
        with _metrics_lock:
            _latencies.append((time.perf_counter() - started) * 1000)
    return response


class TicketIn(BaseModel):
    subject: str = Field(default="", max_length=300, description="Тема обращения, EN/DE")
    body: str = Field(default="", max_length=20_000, description="Текст обращения, EN/DE")


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
    with _metrics_lock:
        stats = _stats.copy()
        queues = dict(_queue_dist)
        durations = sorted(_latencies)
    total = max(stats["requests_total"], 1)
    return {
        "requests_total": stats["requests_total"],
        "errors_4xx": stats["errors_4xx"],
        "errors_5xx": stats["errors_5xx"],
        "error_rate": round((stats["errors_4xx"] + stats["errors_5xx"]) / total, 4),
        "predictions_total": stats["predictions_total"],
        "predicted_queues": queues,
        "rejected_inputs_total": stats["rejected_inputs_total"],
        "unsupported_texts_total": stats["unsupported_texts_total"],
        "secret_requests_total": stats["secret_requests_total"],
        "rate_limited_requests_total": stats["rate_limited_requests_total"],
        "latency_sample_size": len(durations),
        "latency_p95_ms": round(durations[math.ceil(len(durations) * 0.95) - 1], 2) if durations else None,
    }


@app.post("/predict", response_model=PredictOut)
def predict(ticket: TicketIn) -> PredictOut:
    if not ticket.body.strip() and not ticket.subject.strip():
        raise HTTPException(status_code=422, detail="subject и body пусты")
    if model_path() is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    try:
        result = predict_queue(ticket.subject, ticket.body)
    except UnsupportedTextError as exc:
        with _metrics_lock:
            _stats["unsupported_texts_total"] += 1
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except InvalidTicketError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with _metrics_lock:
        _stats["predictions_total"] += 1
        _queue_dist[result["queue"]] += 1
    return PredictOut(queue=result["queue"], confidence=result.get("confidence"))
