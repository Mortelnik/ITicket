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
from ticket_router.security import RequestGuard

ARTIFACT = pathlib.Path("artifacts/model.joblib")


@pytest.fixture(autouse=True)
def reset_request_guard(monkeypatch):
    monkeypatch.setattr(api_module, "_request_guard", RequestGuard())


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


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("Cannot log into my account, error 500", "IT Support"),
        ("Please refund the duplicate charge on my invoice", "Billing and Payments"),
        ("I want to return a damaged product and get a replacement", "Returns and Exchanges"),
    ],
)
def test_saved_model_routes_distinct_topics(client, body, expected):
    """Контроль артефакта: константная модель должна провалить этот тест."""
    response = client.post("/predict", json={"body": body})
    assert response.status_code == 200
    assert response.json() == {"queue": expected, "confidence": None}


@pytest.mark.parametrize("payload", [{}, {"subject": "  "}, {"body": "\n\t"}])
def test_missing_or_blank_text_is_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_subject_without_body_is_supported(client):
    response = client.post("/predict", json={"subject": "Please refund the duplicate charge on my invoice"})
    assert response.status_code == 200
    assert response.json()["queue"] == "Billing and Payments"


@pytest.mark.parametrize("body", ["Верните деньги", "Не могу войти", "VPN не работает", "🦄🦄🦄"])
def test_unsupported_text_is_not_routed(client, body):
    before = client.get("/metrics").json()
    response = client.post("/predict", json={"body": body})
    assert response.status_code == 422
    assert "queue" not in response.json()
    after = client.get("/metrics").json()
    assert after["predictions_total"] == before["predictions_total"]
    assert after["unsupported_texts_total"] == before["unsupported_texts_total"] + 1
    assert after["rejected_inputs_total"] == before["rejected_inputs_total"] + 1


@pytest.mark.parametrize("payload", [{"subject": "x" * 301}, {"body": "x" * 20001}])
def test_oversized_fields_are_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_fields_at_length_limit_are_supported(client):
    response = client.post("/predict", json={"subject": "login " * 50, "body": "invoice " * 2500})
    assert response.status_code == 200


@pytest.mark.parametrize("error", [RuntimeError("inference failure"), ValueError("broken features")])
def test_metrics_count_unhandled_server_error(client, monkeypatch, error):
    def fail(*args):
        raise error

    monkeypatch.setattr(api_module, "predict_queue", fail)
    before = client.get("/metrics").json()
    response = client.post("/predict", json={"body": "login failed"})
    assert response.status_code == 500
    after = client.get("/metrics").json()
    assert after["errors_5xx"] == before["errors_5xx"] + 1
    assert after["predictions_total"] == before["predictions_total"]
    assert after["latency_sample_size"] > before["latency_sample_size"]
    assert after["latency_p95_ms"] >= 0
