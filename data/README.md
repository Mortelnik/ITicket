# Data

Датасет не хранится в git. Для загрузки:

```bash
bash scripts/download_data.sh
```

Скрипт проверяет sha256 (см. `manifest.json`) и падает при несовпадении —
это защита от тихой подмены данных между экспериментами.

- Источник: [Customer IT Support - Ticket Dataset (Kaggle)](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets), автор Tobias Bueck, версия 10.
- Зеркало загрузки: [GitHub CHANDRADEEP2007/feedback-to-backlog-ai-copilot](https://github.com/CHANDRADEEP2007/feedback-to-backlog-ai-copilot) (лицензия CC BY 4.0).
- 28 587 тикетов, 16 колонок; для модели используются `subject`, `body` (признаки) и `queue` (цель).
