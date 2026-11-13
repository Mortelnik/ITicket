"""Кастомные трансформеры признаков."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class CombinedTfidf(TransformerMixin, BaseEstimator):
    """Объединяет словный и символьный TF-IDF в одну разреженную матрицу.

    Оба векторизатора обучаются на train внутри Pipeline — отдельно
    друг от друга они не «видят» val/test.
    """

    def __init__(self, word=None, char=None):
        self.word = word
        self.char = char

    def fit(self, X, y=None):
        self.word.fit(X, y)
        self.char.fit(X, y)
        return self

    def transform(self, X):
        check_is_fitted(self)
        return sparse.hstack([self.word.transform(X), self.char.transform(X)]).tocsr()

    def __sklearn_is_fitted__(self):
        # Проверяем вложенные векторизаторы: это работает и для уже сохранённых
        # артефактов, где у CombinedTfidf нет собственного fitted-атрибута.
        try:
            check_is_fitted(self.word)
            check_is_fitted(self.char)
        except (NotFittedError, TypeError):
            return False
        return True

    def get_feature_names_out(self, input_features=None):
        return np.concatenate(
            [self.word.get_feature_names_out(), self.char.get_feature_names_out()]
        )
