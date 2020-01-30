import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

'''
Replace longitude and latitude with K Means cluster centroids
'''


class LocationCluster(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        coords = X[['longitude', 'latitude']]
        kmeans = KMeans(n_clusters=200).fit(coords)
        centres = kmeans.cluster_centers_
        label_values = np.array([centres[label] for label in kmeans.labels_])
        X[['longitude', 'latitude']] = label_values
        return X
