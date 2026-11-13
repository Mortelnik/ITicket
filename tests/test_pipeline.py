"""Тесты ML-пайплайна: обучение на фикстуре, сохранение/загрузка, инференс."""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from ticket_router.data import combine_text
from ticket_router.experiments import EXPERIMENTS

CLASSES = ["Billing and Payments", "Technical Support", "IT Support"]


def _fixture_df(n_per_class: int = 40) -> pd.DataFrame:
    templates = {
        "Billing and Payments": [
            "invoice payment charged twice refund billing",
            "Rechnung Zahlung doppelt erstattung",
        ],
        "Technical Support": [
            "login failed error code 500 cannot connect server",
            "Anmeldung fehlgeschlagen Fehlercode Server",
        ],
        "IT Support": [
            "vpn setup corporate laptop access internal network",
            "VPN Installation Firmenlaptop internes Netzwerk",
        ],
    }
    rows = []
    for cls, texts in templates.items():
        for i in range(n_per_class):
            t = texts[i % 2]
            rows.append(
                {
                    "subject": f"ticket {cls[:3]} {i}",
                    "body": f"{t} case number {i}",
                    "queue": cls,
                    "language": "de" if i % 2 else "en",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def trained_models():
    """Обучает все конфигурации на синтетической фикстуре (быстро)."""
    df = _fixture_df()
    X, y = combine_text(df), df["queue"]
    return {name: spec["make"]().fit(X, y) for name, spec in EXPERIMENTS.items()}, X, y


def test_all_experiments_fit_and_predict(trained_models):
    models, X, y = trained_models
    for name, model in models.items():
        preds = model.predict(X)
        assert len(preds) == len(y), name
        # На обучающей фикстуре все модели (кроме baseline) должны выучить шаблоны
        if name != "baseline_dummy":
            assert f1_score(y, preds, average="macro") > 0.9, name


def test_baseline_predicts_most_frequent(trained_models):
    models, X, y = trained_models
    preds = models["baseline_dummy"].predict(X)
    assert set(preds) == {"Billing and Payments"}


def test_save_load_roundtrip(trained_models, tmp_path):
    models, X, y = trained_models
    model = models["tfidf_wordchar_svc"]
    before = model.predict(X)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    after = loaded.predict(X)
    assert (before == after).all()


def test_predict_requires_text():
    from ticket_router.predict import load_model, predict_queue

    with pytest.raises(FileNotFoundError):
        load_model(tmp_only := __import__("pathlib").Path("/nonexistent/model.joblib"))
    # модель из артефактов репозитория, если обучена
    import pathlib

    artifact = pathlib.Path("artifacts/model.joblib")
    if not artifact.exists():
        pytest.skip("артефакт модели не найден — сначала запустите обучение")
    load_model(artifact)
    result = predict_queue("Cannot log in, error 500", "Login fails since yesterday")
    assert "queue" in result
    with pytest.raises(ValueError):
        predict_queue("", "   ")
