from sklearn.base import BaseEstimator, TransformerMixin


class TravelingAloneCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Traveling_alone'] = (X['SibSp'] + X['Parch']) == 0
        return X
