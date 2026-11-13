"""Тесты загрузки и валидации данных."""

from __future__ import annotations

import pandas as pd
import pytest

from ticket_router.data import (
    DataValidationError,
    combine_text,
    make_splits,
    validate,
)


def _mini_df():
    return pd.DataFrame(
        {
            "subject": ["Login failed", None, "Rechnung", "Password reset"],
            "body": [
                "Cannot log into my account since yesterday",
                "Zahlung wurde doppelt abgebucht",
                "Wie kann ich meine Rechnung einsehen?",
                "Need help resetting password",
            ],
            "queue": [
                "Technical Support",
                "Billing and Payments",
                "Billing and Payments",
                "Technical Support",
            ],
            "language": ["en", "de", "de", "en"],
        }
    )


def test_validate_ok():
    problems = validate(_mini_df())
    assert problems == []


def test_validate_missing_column():
    df = _mini_df().drop(columns=["queue"])
    with pytest.raises(DataValidationError):
        validate(df)


def test_validate_empty_target():
    df = _mini_df()
    df.loc[0, "queue"] = None
    with pytest.raises(DataValidationError):
        validate(df)


def test_validate_detects_empty_text():
    df = _mini_df()
    df.loc[0, ["subject", "body"]] = ""
    problems = validate(df)
    assert any("пустых текстов" in p for p in problems)


def test_combine_text_fills_missing_subject():
    df = _mini_df()
    texts = combine_text(df)
    # subject отсутствует -> текст не пустой, тело сохранено
    assert "Zahlung" in texts.iloc[1]
    assert texts.str.len().min() > 0


def test_splits_are_disjoint_and_cover_all():
    # два экземпляра каждого класса — минимум для стратифицированного split
    df = pd.concat([_mini_df(), _mini_df()], ignore_index=True)
    train_df, val_df, test_df = make_splits(df, test_size=0.25, val_size=0.25)
    idx = set(train_df.index) | set(val_df.index) | set(test_df.index)
    assert idx == set(df.index)
    assert not (set(train_df.index) & set(val_df.index))
    assert not (set(train_df.index) & set(test_df.index))
    assert not (set(val_df.index) & set(test_df.index))


@pytest.mark.parametrize("column", ["queue", "language"])
def test_validate_rejects_blank_label_or_language(column):
    df = _mini_df()
    df.loc[0, column] = "  "
    with pytest.raises(DataValidationError):
        validate(df)


def test_dataset_warnings_are_not_silently_ignored(tmp_path):
    from ticket_router.data import load_dataset

    df = _mini_df()
    df.loc[0, ["subject", "body"]] = ""
    path = tmp_path / "tickets.csv"
    df.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="пустых текстов"):
        assert len(load_dataset(path)) == len(df)
