"""Fit once on the training fold and transform all other folds."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import SplitData
from .utils import root_path


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    test_ids: pd.Series
    preprocessor: ColumnTransformer
    feature_columns: list[str]


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    numeric = x_train.select_dtypes(include="number").columns.tolist()
    categorical = x_train.select_dtypes(exclude="number").columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        sparse_threshold=0,
    )


def _float32_finite(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Preprocessed features must be a finite 2D float32 array")
    return array


def prepare_data(split: SplitData) -> PreparedData:
    columns = split.x_train.columns.tolist()
    if "Id" in columns or "SalePrice" in columns:
        raise ValueError("Id and SalePrice must not be features")
    preprocessor = build_preprocessor(split.x_train)
    x_train = _float32_finite(preprocessor.fit_transform(split.x_train))
    x_val = _float32_finite(preprocessor.transform(split.x_val.loc[:, columns]))
    x_test = _float32_finite(preprocessor.transform(split.x_test.loc[:, columns]))
    if not x_train.shape[1] == x_val.shape[1] == x_test.shape[1]:
        raise ValueError("Transformed feature widths differ")
    return PreparedData(x_train, x_val, x_test,
                        np.asarray(split.y_train, dtype=np.float32).reshape(-1, 1),
                        np.asarray(split.y_val, dtype=np.float32).reshape(-1, 1),
                        split.test_ids, preprocessor, columns)


def save_preprocessor(prepared: PreparedData, path: str | Path) -> Path:
    target = root_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"preprocessor": prepared.preprocessor,
                 "feature_columns": prepared.feature_columns}, target)
    return target


def load_preprocessor(path: str | Path) -> dict:
    bundle = joblib.load(root_path(path))
    if not {"preprocessor", "feature_columns"}.issubset(bundle):
        raise ValueError("Invalid preprocessor bundle")
    return bundle


def transform_test(test: pd.DataFrame, bundle: dict) -> np.ndarray:
    columns = bundle["feature_columns"]
    if set(columns) != set(test.columns) - {"Id"}:
        raise ValueError("Test feature schema differs from training")
    return _float32_finite(bundle["preprocessor"].transform(test.loc[:, columns]))
