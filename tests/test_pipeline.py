"""Тесты ML-пайплайна: обучение на фикстуре, сохранение/загрузка, инференс."""

from __future__ import annotations

import joblib
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from ticket_router.data import combine_text
from ticket_router.experiments import EXPERIMENTS
from ticket_router.features import CombinedTfidf

CLASSES = ["Billing and Payments", "Technical Support", "IT Support"]


@pytest.mark.parametrize("fitted_parts", [(), ("word",), ("char",)])
def test_combined_tfidf_requires_both_vectorizers_fitted(fitted_parts):
    vectorizer = CombinedTfidf(word=TfidfVectorizer(), char=TfidfVectorizer(analyzer="char"))
    texts = ["invoice payment", "login failed"]
    for name in fitted_parts:
        getattr(vectorizer, name).fit(texts)
    with pytest.raises(NotFittedError):
        check_is_fitted(vectorizer)
    with pytest.raises(NotFittedError):
        Pipeline([("vectorizer", vectorizer)]).transform(texts)


def test_combined_tfidf_loaded_state_supports_pipeline_transform(tmp_path):
    # Старые артефакты содержат обученные вложенные векторизаторы,
    # но не содержат отдельный флаг обучения CombinedTfidf.
    texts = ["invoice payment", "login failed"]
    vectorizer = CombinedTfidf(
        word=TfidfVectorizer().fit(texts),
        char=TfidfVectorizer(analyzer="char").fit(texts),
    )
    path = tmp_path / "vectorizer.joblib"
    joblib.dump(vectorizer, path)
    loaded = joblib.load(path)
    check_is_fitted(loaded)
    expected = vectorizer.transform(texts)
    actual = Pipeline([("vectorizer", loaded)]).transform(texts)
    assert actual.shape == expected.shape
    assert (actual != expected).nnz == 0


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
                    "subject": f"Support request {i}",
                    "body": f"{t} case number {i}",
                    "queue": cls,
                    "language": "de" if i % 2 else "en",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
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
        load_model(__import__("pathlib").Path("/nonexistent/model.joblib"))
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


def test_models_generalize_to_separate_examples(trained_models):
    models, _, _ = trained_models
    # Отдельные обращения, которых нет в обучающей фикстуре; имя класса
    # не подставляется в тему, поэтому проверяется сигнал из текста.
    texts = [
        "Please refund the invoice, I was charged twice",
        "Zahlung doppelt, bitte erstattung",
        "Cannot connect to server, login failed with error code 500",
        "Anmeldung fehlgeschlagen, Fehlercode Server",
        "Need vpn setup on my corporate laptop for internal network access",
        "VPN Installation auf Firmenlaptop, internes Netzwerk",
    ]
    expected = [CLASSES[0], CLASSES[0], CLASSES[1], CLASSES[1], CLASSES[2], CLASSES[2]]
    for name, model in models.items():
        if name != "baseline_dummy":
            assert f1_score(expected, model.predict(texts), average="macro") >= 0.8, name


def test_failed_model_reload_clears_readiness(monkeypatch, tmp_path, trained_models):
    from ticket_router import predict

    monkeypatch.setattr(predict, "_MODEL", trained_models[0]["tfidf_wordchar_svc"])
    monkeypatch.setattr(predict, "_MODEL_PATH", tmp_path / "old.joblib")
    with pytest.raises(FileNotFoundError):
        predict.load_model(tmp_path / "missing.joblib")
    assert predict.model_path() is None
    with pytest.raises(predict.NotTrainedError):
        predict.predict_queue("login", "failed")


def test_model_path_can_be_configured(monkeypatch, tmp_path, trained_models):
    from ticket_router import predict

    artifact = tmp_path / "model.joblib"
    joblib.dump(trained_models[0]["tfidf_wordchar_svc"], artifact)
    monkeypatch.setenv("TICKET_ROUTER_MODEL_PATH", str(artifact))
    monkeypatch.setattr(predict, "_MODEL", None)
    monkeypatch.setattr(predict, "_MODEL_PATH", None)
    predict.load_model()
    assert predict.model_path() == artifact.resolve()
    assert predict.predict_queue("", "vpn setup corporate laptop access internal network")["queue"] == "IT Support"


@pytest.mark.parametrize("name", ["tfidf_wordchar_logreg", "tfidf_wordchar_svc", "tfidf_wordchar_svc_tuned"])
def test_inference_matches_saved_pipeline(monkeypatch, trained_models, name):
    from ticket_router import predict

    model = trained_models[0][name]
    monkeypatch.setattr(predict, "_MODEL", model)
    text = "invoice payment charged twice refund billing"
    result = predict.predict_queue("", text)
    assert result["queue"] == model.predict([text])[0]
    if hasattr(model, "predict_proba"):
        assert result["confidence"] == round(float(model.predict_proba([text])[0].max()), 4)
    else:
        assert "confidence" not in result
