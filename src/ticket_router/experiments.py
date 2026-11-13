"""Определение экспериментов: единый реестр конфигураций моделей.

Все модели — Pipeline(sklearn): векторизация обучается только на train,
поэтому утечки между выборками нет. Фиксированный список даёт
воспроизводимость: один и тот же код -> один и тот же набор запусков.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from .features import CombinedTfidf

EXPERIMENTS: dict[str, dict] = {
    # Эксперимент 0: базовая линия — самый частый класс.
    "baseline_dummy": {
        "description": "DummyClassifier(most_frequent) — нижняя граница качества",
        "make": lambda: DummyClassifier(strategy="most_frequent"),
    },
    # Эксперимент 1: словный TF-IDF + логистическая регрессия.
    "tfidf_word_logreg": {
        "description": "TF-IDF (слова, 1-2 граммы) + LogisticRegression",
        "make": lambda: _pipeline(
            TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), min_df=2,
                max_features=100_000, sublinear_tf=True,
            ),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
    },
    # Эксперимент 2: слова + символьные n-граммы (полезно для опечаток и кодов ошибок).
    "tfidf_wordchar_logreg": {
        "description": "TF-IDF (слова + символьные 3-5 граммы) + LogisticRegression",
        "make": lambda: _pipeline(
            CombinedTfidf(
                word=TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=2,
                    max_features=100_000, sublinear_tf=True,
                ),
                char=TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                    max_features=200_000, sublinear_tf=True,
                ),
            ),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
    },
    # Эксперимент 3: линейный SVM — часто сильнее логрегрессии на разреженных текстах.
    "tfidf_wordchar_svc": {
        "description": "TF-IDF (слова + символьные 3-5 граммы) + LinearSVC",
        "make": lambda: _pipeline(
            CombinedTfidf(
                word=TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2), min_df=2,
                    max_features=100_000, sublinear_tf=True,
                ),
                char=TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                    max_features=200_000, sublinear_tf=True,
                ),
            ),
            LinearSVC(class_weight="balanced", random_state=42),
        ),
    },
    # Эксперимент 5: подбор C для LinearSVC через кросс-валидацию НА TRAIN.
    # Схема честности сохраняется: GridSearchCV (cv=3) видит только train,
    # выбор конфигурации — по val, тест остаётся один раз в конце.
    "tfidf_wordchar_svc_tuned": {
        "description": "TF-IDF (слова+символы) + LinearSVC, GridSearchCV(cv=3) по C",
        "make": lambda: GridSearchCV(
            estimator=_pipeline(
                CombinedTfidf(
                    word=TfidfVectorizer(
                        analyzer="word", ngram_range=(1, 2), min_df=2,
                        max_features=100_000, sublinear_tf=True,
                    ),
                    char=TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                        max_features=200_000, sublinear_tf=True,
                    ),
                ),
                LinearSVC(class_weight="balanced", random_state=42),
            ),
            param_grid={"classifier__C": [1.0, 5.0, 10.0, 20.0]},
            scoring="f1_macro",
            cv=3,
            n_jobs=-1,
            refit=True,
        ),
    },
}


def _pipeline(vectorizer, classifier):
    from sklearn.pipeline import Pipeline

    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])
