"""Кастомные трансформеры признаков."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin


class CombinedTfidf(BaseEstimator, TransformerMixin):
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
        return sparse.hstack([self.word.transform(X), self.char.transform(X)]).tocsr()

    def get_feature_names_out(self, input_features=None):
        return np.concatenate(
            [self.word.get_feature_names_out(), self.char.get_feature_names_out()]
        )
