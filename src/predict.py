"""Load final artifacts, predict prices, and validate the Kaggle submission."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import read_test
from .evaluate import predict_loader
from .model import HousePriceMLP
from .preprocess import load_preprocessor, transform_test
from .utils import choose_device, root_path


def validate_submission(submission: pd.DataFrame, expected_ids: pd.Series) -> None:
    if submission.columns.tolist() != ["Id", "SalePrice"]:
        raise ValueError("Submission columns must be exactly Id,SalePrice")
    if len(submission) != len(expected_ids):
        raise ValueError("Submission row count differs from test.csv")
    if not submission["Id"].reset_index(drop=True).equals(expected_ids.reset_index(drop=True)):
        raise ValueError("Submission Id order differs from test.csv")
    prices = submission["SalePrice"].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or (prices < 0).any():
        raise ValueError("Submission prices must be finite and nonnegative")


def make_submission(checkpoint_path: str | Path = "models/final/best_mlp.pt",
                    output_path: str | Path = "outputs/submission.csv") -> dict:
    checkpoint = torch.load(root_path(checkpoint_path), map_location="cpu", weights_only=True)
    bundle = load_preprocessor(checkpoint["preprocessor_path"])
    test = read_test()
    x_test = transform_test(test, bundle)
    if x_test.shape[1] != checkpoint["input_dim"]:
        raise ValueError("Checkpoint input dimension differs from preprocessor output")
    model = HousePriceMLP(checkpoint["input_dim"], checkpoint["hidden_dims"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    device, reason = choose_device()
    model.to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_test)), batch_size=32, shuffle=False)
    pred_log = predict_loader(model, loader, device)
    if len(pred_log) != len(test) or not np.isfinite(pred_log).all():
        raise ValueError("Log predictions are not finite or have wrong shape")
    prices = np.expm1(pred_log.astype(np.float64))
    if not np.isfinite(prices).all():
        raise ValueError("Price predictions contain NaN or inf")
    clipped = int((prices < 0).sum())
    prices = np.maximum(prices, 0)
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": prices})
    validate_submission(submission, test["Id"])
    target = root_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(target, index=False)
    return {"rows": len(submission), "clipped_negative": clipped, "path": str(target),
            "device": str(device), "device_reason": reason,
            "best_epoch": checkpoint["best_epoch"], "best_val_rmse": checkpoint["best_val_rmse"]}
