from sklearn.base import BaseEstimator, TransformerMixin

'''
Maps 'NA' values to 'None'.
While reading CSV file, pandas converts 'NA' values (Not Applicable)
to NaN.
'''


class NaEnumMapper(BaseEstimator, TransformerMixin):
    def __init__(self, col_names):
        self.col_names = col_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.col_names].fillna('None')
