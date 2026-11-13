# Сервис классификации обращений в поддержку
Учебный проект студентов УрФУ ПИИ, участники:
Шелест М.
Иванова Ю.
Буянов Е.

Система получает текст обращения, определяет его тему — например, «оплата», «доступ к курсу», «техническая проблема» — и направляет обращение в нужную очередь. В проекте можно показать обработку текста, API, тестирование пограничных случаев, мониторинг качества и ограничения автоматической классификации.

**Результат:** test macro-F1 = **0.619** (baseline — 0.045), 10 классов,
языки EN/DE. Решение — скромное по качеству, но честное: данные синтетические,
классы пересекаются семантически (подробности — в отчёте). Есть веб-демо
с имитацией чата (подход заимствован у проекта коллег).

## Структура репозитория

```text
README.md                      ← этот файл
pyproject.toml, uv.lock        ← проект и зафиксированные версии (uv)
Dockerfile                    ← контейнеризация сервиса
src/ticket_router/
  data.py                     ← валидация данных, стратифицированный split
  features.py                 ← CombinedTfidf (слова + символьные n-граммы)
  experiments.py              ← реестр 4 конфигураций моделей
  train.py                    ← обучение, сравнение, финальный тест
  predict.py                  ← инференс
  api.py                      ← FastAPI: /predict, /health, /ready, /metrics
  webui.py                    ← веб-демо с имитацией чата (GET /)
  cli.py                      ← CLI-клиент
tests/                        ← 17 тестов (данные, пайплайн, API, веб-демо)
experiments/                  ← results.csv, splits.json, метрики, матрица ошибок
artifacts/                    ← model.joblib + model_metadata.json
docs/                         ← architecture, api, test_plan, deployment_monitoring, ai_usage, report
data/                         ← manifest.json (CSV не в git)
scripts/download_data.sh      ← загрузка данных с проверкой sha256
.github/workflows/tests.yml    ← CI
```

## Быстрый старт

```bash
# 1. Зависимости (воспроизводимо из uv.lock)
uv sync --locked

# 2. Данные (28 587 тикетов, проверка sha256)
bash scripts/download_data.sh

# 3. Обучение и сравнение 4 экспериментов (~2 мин на CPU)
uv run ticket-router-train

# 4. Тесты
uv run pytest -q

# 5. Сервис (веб-демо откроется на http://localhost:8000/)
uv run uvicorn ticket_router.api:app --port 8000
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' \
  -d '{"subject":"Login failed","body":"Cannot log into my account, error 500"}'
# → {"queue":"IT Support","confidence":null}

# CLI-вариант
uv run ticket-router "Login failed" "Cannot log into my account, error 500"
```

Артефакт `artifacts/model.joblib` уже в репозитории — шаги 2–3 можно пропустить
и сразу запускать API и тесты.

## Эксперименты (фиксированный split 70/15/15, seed=42)

| Эксперимент | val macro-F1 | test macro-F1 |
|---|---|---|
| DummyClassifier (baseline) | 0.045 | — |
| TF-IDF слова + LogisticRegression | 0.469 | — |
| TF-IDF слова+символы + LogisticRegression | 0.491 | — |
| TF-IDF слова+символы + LinearSVC | 0.604 | — |
| **+ GridSearchCV(cv=3) по C, лучший C=10** | **0.619** | **0.619** |

Качество по языкам (test): EN 0.702, DE 0.492. Подробные метрики по классам и
матрица ошибок — в `experiments/`.

## Ключевые решения

- признаки — только `subject` + `body`: колонки `answer/type/priority/tag_*`
  недоступны в момент маршрутизации (исключена утечка данных);
- единый sklearn-pipeline сохраняется одним артефактом (предобработка и модель
  не расходятся между обучением и инференсом);
- выбор модели по val, тест — один раз после выбора;
- `class_weight="balanced"` из-за дисбаланса классов (8362 vs 405);
- основной критерий — macro-F1 (равное внимание редким очередям).

## Документация

- [Отчёт (5–7 стр.)](docs/report.md)
- [Архитектура](docs/architecture.md) · [Контракт API](docs/api.md)
- [Тест-план и найденные проблемы](docs/test_plan.md)
- [Деплой и мониторинг](docs/deployment_monitoring.md)
- [Использование ИИ-ассистентов](docs/ai_usage.md)

## Роли участников

| Участник | Роль | Вклад |
|---|---|---|
| Участник 1 | ML-инженер | EDA, эксперименты, выбор модели |
| Участник 2 | Backend | API, тесты, CI |
| Участник 3 | MLOps | деплой, мониторинг, воспроизводимость |
| Участник 4 | Документация | отчёт, презентация, защита |

Заполните имена участников перед сдачей.

## Ограничения (кратко)

Датасет синтетический; классы поддержки семантически пересекаются (Technical vs
Product vs IT Support дают основную массу ошибок); качество на немецком заметно
ниже; модель не калибрована (LinearSVC без вероятностей). Полный список — в
[отчёте](docs/report.md).

## Источник данных

[Customer IT Support - Ticket Dataset](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
(Tobias Bueck, Kaggle, версия 10, CC BY 4.0), 28 587 тикетов; загружается из
[зеркала](https://github.com/CHANDRADEEP2007/feedback-to-backlog-ai-copilot) с
проверкой sha256.
