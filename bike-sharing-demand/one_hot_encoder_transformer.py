import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class OneHotEncoderTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns) -> None:
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = pd.get_dummies(X, columns=self.columns)
        return X
