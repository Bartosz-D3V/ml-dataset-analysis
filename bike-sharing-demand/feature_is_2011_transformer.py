from sklearn.base import BaseEstimator, TransformerMixin


class FeatureIs2011Transformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['is_2011'] = X['year'] \
                           .apply(lambda datetime: datetime == 2011) * 1
        return X
