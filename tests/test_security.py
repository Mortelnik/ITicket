"""Проверки обходов фильтра, квот, истечения окна и ограниченной памяти."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from ticket_router import api, security
from ticket_router.security import RequestGuard, is_secret_path


@pytest.mark.parametrize("path", [
    "/.env", "/.env.production", "/.git/config", "/.aws/credentials",
    "/.ssh/id_rsa", "/backup/secrets.json", "/credentials.json",
    "/private/server.key", "/.npmrc", "/wp-config.php.bak",
    "/%252eENV", "/%2eGiT/config", "/backup%5c.env", "/.env;download",
])
def test_secret_paths(path):
    assert is_secret_path(path)


@pytest.mark.parametrize("path", ["/", "/predict", "/health", "/ready", "/metrics", "/docs", "/openapi.json"])
def test_public_paths(path):
    assert not is_secret_path(path)


def test_sliding_window_and_independent_clients(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    guard = RequestGuard(limit=2, window=60)
    assert guard.retry_after("a") == 0
    now[0] = 110
    assert guard.retry_after("a") == 0
    assert guard.retry_after("a") == 50
    assert guard.retry_after("b") == 0
    now[0] = 160
    assert guard.retry_after("a") == 0
    assert guard.retry_after("a") == 10


def test_capacity_does_not_evict_active_clients(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(security.time, "monotonic", lambda: now[0])
    guard = RequestGuard(limit=1, window=60, max_clients=2)
    assert guard.retry_after("a") == guard.retry_after("b") == 0
    assert guard.retry_after("c") == 60
    assert guard.retry_after("a") == 60
    assert len(guard._clients) == 2
    now[0] = 160
    assert guard.retry_after("c") == 0
    assert len(guard._clients) == 1


def test_concurrent_requests_cannot_exceed_quota():
    guard = RequestGuard(limit=5)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(guard.retry_after, ["a"] * 30))
    assert results.count(0) == 5


def test_guard_runs_before_routes_and_counts_rejections(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: None)
    monkeypatch.setattr(api, "_request_guard", RequestGuard(limit=2))
    with TestClient(api.app) as client:
        secret_before = api._stats["secret_requests_total"]
        limited_before = api._stats["rate_limited_requests_total"]
        errors_before = api._stats["errors_4xx"]
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200
        # Заголовок от клиента не позволяет получить новую квоту.
        response = client.get("/health", headers={"X-Forwarded-For": "198.51.100.1"})
        assert response.status_code == 429
        assert 1 <= int(response.headers["Retry-After"]) <= 60
        # Фильтр секретов имеет приоритет даже после исчерпания квоты.
        for method in (client.get, client.head, client.post):
            response = method("/%252eenv.production")
            assert response.status_code == 444
            assert response.content == b""
        assert api._stats["secret_requests_total"] == secret_before + 3
        assert api._stats["rate_limited_requests_total"] == limited_before + 1
        assert api._stats["errors_4xx"] == errors_before + 4


@pytest.mark.parametrize("setting", ["limit", "window", "max_clients"])
def test_invalid_settings_fail_at_startup(setting):
    with pytest.raises(ValueError):
        RequestGuard(**{setting: 0})
