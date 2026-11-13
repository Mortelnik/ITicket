# Образ сервиса ticket-router.
# Сборка контейнера в среде разработки НЕ выполнялась (Docker-демон недоступен);
# маршрут внутри контейнера идентичен проверенному локальному запуску.
# Первая итерация команды: docker build . && docker run -p 8000:8000 + smoke-тест.

FROM python:3.12-slim

# uv для установки зависимостей из lock-файля (воспроизводимость)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Сначала только манифест зависимостей — кэш слоя
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

# Код проекта
COPY src ./src
COPY README.md ./

# Устанавливаем сам проект
RUN uv sync --locked --no-dev

# Артефакт модели кладётся в образ на этапе сборки.
# В реальном пайплайне модель передаётся через registry (S3/MLflow), а не запекается в образ.
COPY artifacts/model.joblib artifacts/model_metadata.json ./artifacts/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"

CMD ["uvicorn", "ticket_router.api:app", "--host", "0.0.0.0", "--port", "8000"]
