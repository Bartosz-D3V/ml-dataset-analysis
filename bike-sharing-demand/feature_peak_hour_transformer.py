from sklearn.base import BaseEstimator, TransformerMixin


class FeaturePeakHourTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['peak_hour'] = X['hour'] \
                             .apply(lambda datetime: datetime in (7, 8, 17, 18)) * 1
        return X
