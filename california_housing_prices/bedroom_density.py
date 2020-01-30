from sklearn.base import BaseEstimator, TransformerMixin

'''
Add new feature - proportion of bedrooms to rooms
'''


class BedroomDensity(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['bedroom_density'] = X['total_bedrooms'] / X['total_rooms']
        return X
