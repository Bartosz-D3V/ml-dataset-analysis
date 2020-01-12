from sklearn.base import BaseEstimator, TransformerMixin


class RelativesColumnCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Relatives'] = X['SibSp'] + X['Parch']
        return X.drop(columns=['SibSp', 'Parch'])
