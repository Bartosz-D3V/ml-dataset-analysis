from sklearn.base import BaseEstimator, TransformerMixin


class LabelRemover(BaseEstimator, TransformerMixin):
    def __init__(self, label):
        self.label = label

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.label)
