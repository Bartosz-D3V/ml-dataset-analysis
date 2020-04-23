import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['iso_datetime'] = pd.to_datetime(X['datetime'])
        X['day'] = X['iso_datetime'].apply(lambda datetime: datetime.strftime('%d')).astype("int")
        X['weekday'] = X['iso_datetime'].apply(lambda datetime: datetime.strftime('%w')).astype("int")
        X['month'] = X['iso_datetime'].apply(lambda datetime: datetime.strftime('%m')).astype("int")
        X['year'] = X['iso_datetime'].apply(lambda datetime: datetime.strftime('%Y')).astype("int")
        X['hour'] = X['iso_datetime'].apply(lambda datetime: datetime.strftime('%H')).astype("int")
        return X
