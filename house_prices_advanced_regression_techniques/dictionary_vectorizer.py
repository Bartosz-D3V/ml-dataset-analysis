from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer

'''
Converts category columns into matrix
'''


class DictionaryVectorizer(BaseEstimator, TransformerMixin):
    _feature_names = list()
    _dict_vectorizer = None

    def fit(self, X, y=None):
        self._dict_vectorizer = DictVectorizer(sparse=False)
        x_dict = X.to_dict(orient='records')
        self._dict_vectorizer.fit(x_dict)
        self._feature_names = self._dict_vectorizer.get_feature_names()

    def transform(self, X, y=None):
        x_dict = X.to_dict(orient='records')
        return self._dict_vectorizer.transform(x_dict)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)

    def get_feature_names(self):
        return self._feature_names
