from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

'''
Converts category columns into matrix
'''


class DictionaryVectorizer(BaseEstimator, TransformerMixin):

    def __init__(self) -> None:
        self.dict_vectorizer = None
        self.feature_names = []

    def fit(self, X, y=None):
        self.dict_vectorizer = DictVectorizer(sparse=False)
        x_dict = X.to_dict(orient='records')
        self.dict_vectorizer.fit(x_dict)
        self.feature_names = self.dict_vectorizer.get_feature_names()

    def transform(self, X, y=None):
        x_dict = X.to_dict(orient='records')
        return self.dict_vectorizer.transform(x_dict)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names(self):
        return self.feature_names
