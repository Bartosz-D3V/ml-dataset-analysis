from sklearn.base import BaseEstimator, TransformerMixin

THRESHOLD = .1


class NumericalCleaner(BaseEstimator, TransformerMixin):
    _top_features = []

    def fit(self, X, y=None):
        x_labelled = X
        x_labelled['SalePrice'] = y
        features_corr = x_labelled.corr()
        price_corr = features_corr['SalePrice'].sort_values(ascending=False).drop('SalePrice')
        self._top_features = price_corr[price_corr > THRESHOLD]

    def transform(self, X, y=None):
        return X[self._top_features.keys()]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)
