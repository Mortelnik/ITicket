"""Тесты REST API через TestClient.

Тесты устанавливают собственное состояние модели (фикстуры), а не зависят
от наличия артефакта в репозитории и порядка запуска.
"""

from __future__ import annotations

import pathlib

import pytest
from fastapi.testclient import TestClient

from ticket_router import api as api_module
from ticket_router import predict as predict_module
from ticket_router.api import app

ARTIFACT = pathlib.Path("artifacts/model.joblib")


@pytest.fixture
def client():
    # Контекстный менеджер нужен, чтобы сработал lifespan (загрузка модели).
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def client_no_model(monkeypatch):
    """Клиент с незагруженной моделью: имитирует запуск без артефакта.

    lifespan при старте вызывает load_model() — подменяем её на no-op,
    иначе она перезатрёт monkeypatch состояния модуля predict.
    """
    monkeypatch.setattr(api_module, "load_model", lambda: None)
    monkeypatch.setattr(predict_module, "_MODEL", None)
    monkeypatch.setattr(predict_module, "_MODEL_PATH", None)
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert "model_loaded" in resp.json()


def test_predict_requires_body_or_subject(client):
    resp = client.post("/predict", json={"subject": "", "body": ""})
    assert resp.status_code == 422


def test_predict_returns_queue(client):
    if not ARTIFACT.exists():
        pytest.skip("артефакт модели не найден — сначала запустите обучение")
    resp = client.post(
        "/predict",
        json={"subject": "Login failed", "body": "Cannot log into my account, error 500"},
    )
    assert resp.status_code == 200
    assert "queue" in resp.json()


def test_ready_503_without_model(client_no_model):
    resp = client_no_model.get("/ready")
    assert resp.status_code == 503


def test_predict_503_without_model(client_no_model):
    resp = client_no_model.post("/predict", json={"subject": "x", "body": "y"})
    assert resp.status_code == 503


def test_metrics_counts_errors(client):
    if not ARTIFACT.exists():
        pytest.skip("артефакт модели не найден — сначала запустите обучение")
    before = client.get("/metrics").json()
    client.post("/predict", json={"subject": "x", "body": "refund my invoice payment"})
    client.post("/predict", json={"subject": "", "body": ""})  # 422
    after = client.get("/metrics").json()
    assert after["predictions_total"] == before["predictions_total"] + 1
    assert after["errors_4xx"] >= before["errors_4xx"] + 1
    assert after["requests_total"] > before["requests_total"]


def test_demo_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ticket-router" in resp.text
    # страница обращается к тому же эндпоинту предсказания
    assert "/predict" in resp.text
