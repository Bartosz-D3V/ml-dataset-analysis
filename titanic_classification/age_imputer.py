import logging

from datawig import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class AgeImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logging.getLogger().setLevel(logging.CRITICAL)
        if X['Age'].isna().sum() == 0:
            return X
        df_train = X[X['Age'].notnull()]
        df_test = X[X['Age'].isnull()]
        imputer = SimpleImputer(
            input_columns=['Title', 'Pclass', 'Sex', 'Parch', 'SibSp'],
            output_column='Age',
            output_path='age_imputer_model'
        )
        imputer.fit(train_df=df_train, num_epochs=100)
        missing_ages = imputer.predict(df_test)['Age_imputed'] \
            .apply(lambda age: int(age)) \
            .apply(lambda age: 0 if age < 0 else age)
        X['Age'] = X['Age'].fillna(missing_ages)
        return X
