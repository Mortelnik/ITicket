"""Загрузка, валидация и разбиение данных.

Используются только колонки subject + body (вход модели) и queue (цель).
Колонки answer, type, priority, tag_* исключены: в момент маршрутизации
нового обращения они ещё не существуют — их использование привело бы
к утечке данных (leakage).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "queue"
TEXT_COLUMNS = ["subject", "body"]
RANDOM_STATE = 42

REQUIRED_COLUMNS = ["subject", "body", "queue", "language"]


class DataValidationError(ValueError):
    """Ошибка валидации входного датасета."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def combine_text(df: pd.DataFrame) -> pd.Series:
    """subject + body -> единый текст. Пропуски subject заменяются пустой строкой."""
    subject = df["subject"].fillna("").astype(str).str.strip()
    body = df["body"].fillna("").astype(str).str.strip()
    return (subject + "\n" + body).str.strip()


def validate(df: pd.DataFrame) -> list[str]:
    """Проверка схемы и качества данных. Возвращает список найденных проблем."""
    problems: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Отсутствуют колонки: {missing}")
    if df.empty:
        raise DataValidationError("Датасет пуст")
    if df[TARGET].isna().any():
        raise DataValidationError("Есть пропуски в целевой переменной queue")
    n_empty = int((combine_text(df).str.len() == 0).sum())
    if n_empty:
        problems.append(f"пустых текстов (subject+body): {n_empty}")
    n_dup = int(df.duplicated(subset=TEXT_COLUMNS).sum())
    if n_dup:
        problems.append(f"полных дубликатов текста: {n_dup}")
    return problems


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate(df)
    return df


def make_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Стратифицированное разбиение train/val/test по queue.

    Одно разбиение фиксируется для всех экспериментов (сравнимость).
    """
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, random_state=random_state, stratify=df[TARGET]
    )
    relative = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative, random_state=random_state, stratify=temp_df[TARGET]
    )
    return train_df, val_df, test_df


def split_info(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    return {
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "train_queue_dist": train_df[TARGET].value_counts().to_dict(),
        "val_queue_dist": val_df[TARGET].value_counts().to_dict(),
        "test_queue_dist": test_df[TARGET].value_counts().to_dict(),
    }


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> dict:
    """Сохраняет индексы разбиения (не сами данные) для воспроизводимости."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "random_state": RANDOM_STATE,
        "train_idx": train_df.index.tolist(),
        "val_idx": val_df.index.tolist(),
        "test_idx": test_df.index.tolist(),
        **split_info(train_df, val_df, test_df),
    }
    with open(out_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return payload
