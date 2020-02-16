from sklearn.base import BaseEstimator, TransformerMixin


class BirkenheadDrill(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['BirkenheadDrill'] = X.apply(lambda x: 1 if ((x['Sex'] == 'female') or (x['Age'] < 16)) else 0, axis=1)
        return X
