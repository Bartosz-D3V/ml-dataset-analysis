import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

THRESHOLD = .2


class CategoryCleaner(BaseEstimator, TransformerMixin):
    _irrelevant_feature_ranking_indices = []

    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        feature_start_index = np.size(X, 1) - len(self.feature_names)
        feature_indices = list(range(feature_start_index, np.size(X, 1)))
        feature_scores = mutual_info_classif(X[:, feature_indices], y)
        feature_ranking = sorted(zip(feature_scores, self.feature_names, feature_indices), reverse=True)
        relevant_feature_ranking = list(filter(lambda x: x[0] > THRESHOLD, feature_ranking))
        relevant_feature_ranking_indices = [val[2] for val in relevant_feature_ranking]
        irrelevant_feature_ranking_indices = list(
            set(range(feature_start_index, np.size(X, 1))) - set(relevant_feature_ranking_indices))
        self._irrelevant_feature_ranking_indices = irrelevant_feature_ranking_indices

    def transform(self, X):
        return np.delete(X, self._irrelevant_feature_ranking_indices, axis=1)

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X)
