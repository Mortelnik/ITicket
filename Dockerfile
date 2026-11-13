# Образ учебного сервиса ticket-router. Сборка и smoke-тест описаны в CI.

FROM python:3.12-slim

# uv для установки зависимостей из lock-файля (воспроизводимость)
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv

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
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TICKET_ROUTER_MODEL_PATH=/app/artifacts/model.joblib

RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready', timeout=2)"

CMD ["uvicorn", "ticket_router.api:app", "--host", "0.0.0.0", "--port", "8000"]
