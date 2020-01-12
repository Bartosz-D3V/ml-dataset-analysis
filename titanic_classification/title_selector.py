from sklearn.base import BaseEstimator, TransformerMixin


class TitleSelector(BaseEstimator, TransformerMixin):
    title_dictionary = {
        "CAPT": "Officer",
        "MAJOR": "Officer",
        "JONKHEER": "Royalty",
        "DON": "Royalty",
        "THE COUNTESS": "Royalty",
        "SIR": "Royalty",
        "MLLE": "Miss",
        "DR": "Officer",
        "REV": "Officer",
        "MME": "Mrs",
        "MS": "Mrs",
        "COL": "Officer",
        "MR": "Mr",
        "MRS": "Mrs",
        "MISS": "Miss",
        "MASTER": "Master",
        "DONA": "Royalty",
        "LADY": "Royalty",
    }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Title'] = X['Name'] \
            .apply(lambda last_name: last_name.split(',')[1]) \
            .apply(lambda name: name.split('.')[0]) \
            .apply(lambda title: title.strip().upper()) \
            .map(TitleSelector.title_dictionary)
        return X
