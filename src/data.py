"""Read the original Kaggle files and define the fixed holdout."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import root_path


@dataclass
class SplitData:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    test_ids: pd.Series


def read_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    folder = root_path(data_dir)
    opts = {"keep_default_na": False, "na_values": ["", "NA"]}
    train = pd.read_csv(folder / "train.csv", **opts)
    test = read_test(data_dir)
    if "SalePrice" not in train or "SalePrice" in test or "Id" not in train or "Id" not in test:
        raise ValueError("Unexpected House Prices input schema")
    if train["Id"].duplicated().any() or test["Id"].duplicated().any():
        raise ValueError("Id must be unique within each file")
    if train["SalePrice"].isna().any() or (train["SalePrice"] < 0).any():
        raise ValueError("SalePrice must be present and nonnegative")
    features = train.drop(columns=["Id", "SalePrice"]).columns.tolist()
    if set(features) != set(test.drop(columns="Id").columns):
        raise ValueError("Train/test feature schemas differ")
    return train, test


def read_test(data_dir: str | Path = "data") -> pd.DataFrame:
    test = pd.read_csv(root_path(data_dir) / "test.csv",
                       keep_default_na=False, na_values=["", "NA"])
    if "Id" not in test or "SalePrice" in test or test["Id"].duplicated().any():
        raise ValueError("Unexpected test schema or duplicate Id")
    return test


def split_data(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42,
               test_size: float = 0.2) -> SplitData:
    features = train.drop(columns=["Id", "SalePrice"])
    y_log = np.log1p(train["SalePrice"].to_numpy(dtype=np.float64))
    x_train, x_val, y_train, y_val = train_test_split(
        features, y_log, test_size=test_size, random_state=seed
    )
    x_test = test.drop(columns="Id").loc[:, features.columns]
    return SplitData(x_train, x_val, x_test, y_train, y_val, test["Id"].copy())
