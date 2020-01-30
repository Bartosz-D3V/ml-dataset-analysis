from sklearn.base import BaseEstimator, TransformerMixin

'''
Add new feature - proportion of population to median_income
'''


class PopulationValue(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['population_value'] = X['population'] / X['median_income']
        return X
