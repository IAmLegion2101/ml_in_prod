"""help funcs for preparing data"""

import pandas as pd
from sklearn.model_selection import train_test_split
import typing
from src.config import DataSplitParams


def get_splitted_data(df: pd.DataFrame,
                      split_params: DataSplitParams) -> typing.Tuple[pd.DataFrame,
                                                                     pd.DataFrame,
                                                                     pd.DataFrame]:
    """
    Get train, test and val df
    :param df: general df
    :param split_params: params for split
    :return:
    returns tuple of train test and val df
    """
    train, test = train_test_split(df,
                                   test_size=split_params.test_size,
                                   random_state=split_params.random_state)
    train, val = train_test_split(train,
                                  test_size=split_params.val_size,
                                  random_state=split_params.random_state)
    return train, val, test


def get_target_data(df: pd.DataFrame, target: str) -> typing.Tuple[pd.DataFrame,
                                                                   pd.Series]:
    """
    func for getting target feature
    :param df: general df
    :param target: target feature name
    :return:
    returns tuple of df without target feature and target column
    """
    data = df.drop(target, axis=1)
    target_col = df[target]
    return data, target_col
