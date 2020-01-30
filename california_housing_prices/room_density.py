from sklearn.base import BaseEstimator, TransformerMixin

'''
Add new feature - proportion of rooms to households
'''


class RoomDensity(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['room_density'] = X['total_rooms'] / X['households']
        return X
