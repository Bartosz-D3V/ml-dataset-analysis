import os

import pandas as pd
from pandas import DataFrame


def load_data(path: str, filename: str, keep_na=True) -> DataFrame:
    csv_path = os.path.join(path, filename)
    return pd.read_csv(csv_path, keep_default_na=keep_na)
