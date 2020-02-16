import logging

import numpy as np
from datawig import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class DeckImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logging.getLogger().setLevel(logging.CRITICAL)
        X['Deck'] = X['Cabin'].apply(lambda cabin: np.nan if cabin is np.nan else cabin[0])
        df_train = X[X['Cabin'].notnull()]
        df_test = X[X['Cabin'].isnull()]
        imputer = SimpleImputer(
            input_columns=['Title', 'Pclass', 'Fare', 'Age', 'Embarked'],
            output_column='Deck',
            output_path='deck_imputer_model'
        )
        imputer.fit(train_df=df_train, num_epochs=100)
        X['Deck'] = X['Deck'].fillna(imputer.predict(df_test)['Deck_imputed'])
        return X
