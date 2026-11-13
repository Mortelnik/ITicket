"""Проверка запущенного HTTP-сервиса стандартной библиотекой Python."""

from __future__ import annotations

import argparse
import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(base_url: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(base_url.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=5) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--wait-seconds", type=float, default=30)
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            status, _ = request(args.url, "/ready")
            if status == 200:
                break
        except (URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Сервис не стал готов в отведённое время")
        time.sleep(0.5)
    status, health = request(args.url, "/health")
    check(status == 200 and health["model_loaded"], "Модель не загружена")
    _, before = request(args.url, "/metrics")
    cases = [
        ("Cannot log into my account, error 500", "IT Support"),
        ("Please refund the duplicate charge on my invoice", "Billing and Payments"),
        ("I want to return a damaged product and get a replacement", "Returns and Exchanges"),
    ]
    for body, expected in cases:
        status, result = request(args.url, "/predict", {"body": body})
        check(status == 200 and result.get("queue") == expected, f"Неверная маршрутизация: {body}: {result}")
        check(result.get("confidence") is None, "LinearSVC не должна выдавать вероятность")
    for payload in ({}, {"body": "Верните деньги"}, {"body": "🦄🦄🦄"}, {"body": "x" * 20001}):
        status, result = request(args.url, "/predict", payload)
        check(status == 422 and "queue" not in result, f"Некорректный вход не отклонён: {status}")
    status, after = request(args.url, "/metrics")
    check(status == 200, "Метрики недоступны")
    check(after["predictions_total"] == before["predictions_total"] + 3, "Счётчик предсказаний неверен")
    check(after["rejected_inputs_total"] == before["rejected_inputs_total"] + 4, "Счётчик отказов неверен")
    check(after["unsupported_texts_total"] == before["unsupported_texts_total"] + 2, "Счётчик неизвестного текста неверен")
    check(after["latency_sample_size"] > 0 and after["latency_p95_ms"] is not None, "Нет метрики задержки")
    print("OK: readiness, health, 3 очереди, 4 пограничных случая, метрики")


if __name__ == "__main__":
    main()
