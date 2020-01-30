from sklearn.base import BaseEstimator, TransformerMixin

'''
Add new feature - proportion of households to population
'''


class HouseholdDensity(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['household_density'] = X['households'] / X['population']
        return X
