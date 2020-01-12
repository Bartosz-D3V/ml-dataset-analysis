'''
Implementation of SimpleImputer that returns Pandas DataFrame
'''
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


class DataFrameSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fit_params):
        self.simple_imputer = None
        self.fit_params = fit_params

    def fit(self, X, y=None):
        self.simple_imputer = SimpleImputer(**self.fit_params)
        return self

    def transform(self, X, y=None):
        data = self.simple_imputer.fit_transform(X)
        return DataFrame(data, columns=X.columns.values)
