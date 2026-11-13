#!/usr/bin/env bash
# Скачивание датасета Customer IT Support - Ticket Dataset (Kaggle v10).
# Зеркало: репозиторий CHANDRADEEP2007/feedback-to-backlog-ai-copilot
# (перераспространяет версию 10 датасета под CC BY 4.0, см. data/manifest.json).
set -euo pipefail

URL="https://raw.githubusercontent.com/CHANDRADEEP2007/feedback-to-backlog-ai-copilot/main/data/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
OUT="data/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
EXPECTED_SHA256="f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0"

mkdir -p data
curl -sL "$URL" -o "$OUT"

ACTUAL_SHA256=$(sha256sum "$OUT" | cut -d' ' -f1)
if [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
  echo "ОШИБКА: sha256 не совпал." >&2
  echo "ожидается: $EXPECTED_SHA256" >&2
  echo "получен:  $ACTUAL_SHA256" >&2
  rm "$OUT"
  exit 1
fi

echo "OK: $OUT ($ACTUAL_SHA256)"
