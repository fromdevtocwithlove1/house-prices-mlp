"""Reusable descriptive analysis and plots for the real Kaggle data."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .data import read_data
from .utils import root_path


def run_eda(data_dir: str | Path = "data", figures_dir: str | Path = "outputs/figures") -> dict:
    train, test = read_data(data_dir)
    out = root_path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = train["SalePrice"]
    features = train.drop(columns=["Id", "SalePrice"])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    missing = train.isna().sum().sort_values(ascending=False)
    correlations = train[numeric + ["SalePrice"]].corr(numeric_only=True)["SalePrice"].drop("SalePrice").sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(target, bins=40, ax=ax[0]); ax[0].set_title("SalePrice")
    sns.histplot(np.log1p(target), bins=40, ax=ax[1]); ax[1].set_title("log1p(SalePrice)")
    ax[1].set_xlabel("log1p(SalePrice)")
    fig.tight_layout(); fig.savefig(out / "target_histograms.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    missing.head(20).sort_values().plot.barh(ax=ax)
    ax.set_title("Top 20 missing columns (train)"); ax.set_xlabel("Missing rows")
    fig.tight_layout(); fig.savefig(out / "missing_values.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(11, 8))
    for col, ax in zip(["GrLivArea", "TotalBsmtSF", "GarageArea", "YearBuilt"], axs.flat):
        sns.scatterplot(data=train, x=col, y="SalePrice", ax=ax, s=16, alpha=0.6)
    fig.tight_layout(); fig.savefig(out / "numeric_scatter.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=train, x="OverallQual", y="SalePrice", ax=ax)
    fig.tight_layout(); fig.savefig(out / "overallqual_boxplot.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    order = train.groupby("Neighborhood")["SalePrice"].median().sort_values().index
    sns.boxplot(data=train, x="Neighborhood", y="SalePrice", order=order, ax=ax)
    ax.tick_params(axis="x", rotation=75)
    fig.tight_layout(); fig.savefig(out / "neighborhood_boxplot.png", dpi=150); plt.close(fig)

    report = {
        "train_shape": list(train.shape), "test_shape": list(test.shape),
        "train_dtypes": {k: int(v) for k, v in train.dtypes.astype(str).value_counts().items()},
        "test_dtypes": {k: int(v) for k, v in test.dtypes.astype(str).value_counts().items()},
        "train_duplicates": int(train.duplicated().sum()), "test_duplicates": int(test.duplicated().sum()),
        "numeric_count": len(numeric), "categorical_count": len(categorical),
        "numeric_features": numeric, "categorical_features": categorical,
        "target": {"mean": float(target.mean()), "median": float(target.median()), "std": float(target.std()),
                   "skew": float(target.skew()), "log_skew": float(np.log1p(target).skew())},
        "top_missing_train": {k: int(v) for k, v in missing.head(20).items()},
        "top_missing_test": {k: int(v) for k, v in test.isna().sum().sort_values(ascending=False).head(20).items()},
        "top_correlations": {k: float(v) for k, v in correlations.head(15).items()},
        "outlier_candidates": train.loc[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000),
                                        ["Id", "GrLivArea", "SalePrice"]].to_dict("records"),
    }
    return report
