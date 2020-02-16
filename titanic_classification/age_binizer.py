from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


class AgeBinizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        kmeans = KMeans(n_clusters=5)
        X['Age'] = kmeans.fit_predict(X['Age'].to_numpy().reshape(-1, 1))
        return X
