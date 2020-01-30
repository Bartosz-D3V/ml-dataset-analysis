import math

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


'''
Add new feature - distance to the nearest city
'''


class CityDistance(BaseEstimator, TransformerMixin):
    r = 6371

    def __init__(self, cities: DataFrame):
        self.cities = cities

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        d_lat_1 = X['latitude']
        d_lon_1 = X['longitude']
        X['dist_city'] = 0
        for row in self.cities.itertuples():
            d_lat_2 = row.latitude
            d_lon_2 = row.longitude
            d_lat = to_rad(np.subtract(d_lat_2, d_lat_1))
            d_lon = to_rad(np.subtract(d_lon_2, d_lon_1))
            a = np.sin(d_lat / 2) * np.sin(d_lat / 2) + \
                np.cos(to_rad(d_lat_1)) * np.cos(to_rad(d_lat_2)) * \
                np.sin(d_lon / 2) * np.sin(d_lon / 2)
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            d = CityDistance.r * c
            X['dist_city'] = np.where(X['dist_city'] < d, d, X['dist_city'])
        return X


def to_rad(degree):
    return degree * (math.pi / 180)
