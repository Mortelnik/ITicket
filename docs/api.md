# Контракт API

Базовый URL: `http://localhost:8000`

## GET /

Веб-демо с имитацией чата с отделом поддержки (подход заимствован у проекта
ai_ticket_classifier): вводится тема и текст обращения, страница показывает
предсказанный отдел как «ответ поддержки». Обращается к тому же `POST /predict`.

## POST /predict

Классификация одного обращения.

Запрос:

```json
{"subject": "Login failed", "body": "Cannot log into my account, error 500"}
```

Ответ 200:

```json
{"queue": "IT Support", "confidence": null}
```

`confidence` заполнен только для моделей с `predict_proba`; выбранная LinearSVC
вероятности не выдаёт (см. ограничение в architecture.md).

Ошибки:
- `422` — оба поля пусты или нарушена схема запроса;
- `503` — модель не загружена (артефакт отсутствует при старте).

## GET /health

Живость процесса (всегда 200, если процесс отвечает):

```json
{"status": "ok", "model_loaded": true, "version": "0.1.0", "uptime_seconds": 3.6}
```

Используется для liveness-проб. Готовность модели проверяется отдельно.

## GET /ready

Readiness-проверка: `503`, пока артефакт модели не загружен, иначе `200`.
Используется в Docker healthcheck и readiness-пробе Kubernetes: трафик
не направляется на поду без модели.

## GET /metrics

Счётчики запросов и ошибок по классам статусов, распределение предсказанных классов:

```json
{"requests_total": 4, "errors_4xx": 1, "errors_5xx": 0, "error_rate": 0.25,
 "predictions_total": 0, "predicted_queues": {}}
```

В учебной реализации счётчики в памяти процесса и сбрасываются при рестарте;
в production их заменяет Prometheus-экспортер (см. deployment_monitoring.md).

## CLI

```bash
uv run ticket-router "Login failed" "Cannot log into my account, error 500"
# {"queue": "IT Support", "confidence": null}
```
