# Data

Датасет не хранится в git. Для загрузки:

```bash
uv run python scripts/download_data.py
```

Скрипт проверяет sha256 (см. `manifest.json`) и падает при несовпадении —
это защита от тихой подмены данных между экспериментами.
Команда работает на Windows и Linux; уже скачанный файл с верным хэшем
повторно не загружается. Новая загрузка заменяет файл только после проверки
хэша. Скрипт `download_data.sh` сохранён как прежний Linux-вариант.

Аудит схемы и сохранённых выборок:

```bash
uv run python scripts/check_data.py --output experiments/data_validation.json
```

В сохранённом split нет точных текстовых пересечений. Одна пара train/test
совпадает после нормализации регистра и пробелов; это остаточный риск утечки.

- Источник: [Customer IT Support - Ticket Dataset (Kaggle)](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets), автор Tobias Bueck, версия 10.
- Зеркало загрузки: [GitHub CHANDRADEEP2007/feedback-to-backlog-ai-copilot](https://github.com/CHANDRADEEP2007/feedback-to-backlog-ai-copilot) (лицензия CC BY 4.0).
- 28 587 тикетов, 16 колонок; для модели используются `subject`, `body` (признаки) и `queue` (цель).
- Языки: английский и немецкий. Русских обучающих обращений нет.
