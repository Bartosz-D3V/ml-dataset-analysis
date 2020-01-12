from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer


class AgeImputer(BaseEstimator, TransformerMixin):
    def __init__(self, convert_to_bins=False):
        self.simple_imputer = None
        self.convert_to_bins = convert_to_bins

    def fit(self, X, y=None):
        self.simple_imputer = SimpleImputer()
        return self

    def transform(self, X, y=None):
        X['Age'] = X['Age'].fillna(X.groupby(['Title', 'Pclass']).transform('median')['Age'])
        if self.convert_to_bins:
            bins_dsc = KBinsDiscretizer(n_bins=25, encode='ordinal')
            X['Age'] = bins_dsc.fit_transform(X[['Age']])
        return X
