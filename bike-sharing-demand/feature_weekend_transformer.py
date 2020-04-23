import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureWeekendTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['weekend'] = X['iso_datetime'] \
                           .apply(lambda datetime: datetime.strftime('%w')).astype("int") \
                           .apply(lambda datetime: datetime in (0, 6)) * 1
        return X
