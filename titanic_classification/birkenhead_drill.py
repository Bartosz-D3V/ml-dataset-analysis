from sklearn.base import BaseEstimator, TransformerMixin


class BirkenheadDrill(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['BirkenheadDrill'] = X.apply(lambda x: (x['Sex'] == 'female') or (x['Age'] < 18), axis=1)
        return X
