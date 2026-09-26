"""Training, early stopping, artifact storage, and experiment orchestration."""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import load_config
from .data import read_data, split_data
from .dataset import make_loaders
from .evaluate import predict_loader, rmse_log, save_figures
from .model import HousePriceMLP
from .preprocess import PreparedData, prepare_data, save_preprocessor
from .utils import choose_device, root_path, set_seed


EXPERIMENT_COLUMNS = ["experiment_name", "architecture", "dropout", "learning_rate",
                      "weight_decay", "best_epoch", "best_val_rmse"]


def _evaluate(model, loader, device) -> float:
    model.eval()
    squared_error = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            residual = model(x.to(device)) - y.to(device)
            squared_error += residual.square().sum().item()
            count += y.numel()
    return squared_error / count


def _save_row(row: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing.loc[existing.experiment_name != row["experiment_name"]]
        table = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        table = pd.DataFrame([row])
    table = table[EXPERIMENT_COLUMNS].sort_values("experiment_name")
    table.to_csv(path, index=False)


def train_experiment(name: str, prepared: PreparedData, config: dict,
                     artifact_dir: str | Path | None = None,
                     result_name: str | None = None,
                     record_experiment: bool = True) -> dict:
    if name not in config["experiments"]:
        raise KeyError(name)
    set_seed(int(config["seed"]))
    torch.set_num_threads(1)  # Small tabular batches run faster and more consistently here.
    device, reason = choose_device(config.get("device", "auto"))
    params = config["experiments"][name]
    folder = root_path(artifact_dir or f"models/experiments/{name}")
    folder.mkdir(parents=True, exist_ok=True)
    preprocessor_path = save_preprocessor(prepared, folder / "preprocessor.joblib")
    model = HousePriceMLP(prepared.x_train.shape[1], params["hidden_dims"], params["dropout"]).to(device)
    # The target is near 12 in log units; start the intercept there using training data only.
    with torch.no_grad():
        model.net[-1].bias.fill_(float(prepared.y_train.mean()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]),
                                  weight_decay=float(params["weight_decay"]))
    criterion = nn.MSELoss()
    train_loader, val_loader, _, train_eval_loader = make_loaders(
        prepared, int(config["batch_size"]), int(config["seed"]))
    history = []
    best_rmse, best_epoch, stale = float("inf"), 0, 0
    checkpoint_path = folder / "best_mlp.pt"
    print(f"{name} | device={device} | {reason}", flush=True)
    for epoch in range(1, int(config["max_epochs"]) + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        train_loss = _evaluate(model, train_eval_loader, device)
        val_loss = _evaluate(model, val_loader, device)
        train_rmse, val_rmse = float(np.sqrt(train_loss)), float(np.sqrt(val_loss))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "train_rmse": train_rmse, "val_rmse": val_rmse})
        print(f"Epoch {epoch:03d} | Train RMSE: {train_rmse:.6f} | Val RMSE: {val_rmse:.6f}", flush=True)
        if val_rmse < best_rmse:
            best_rmse, best_epoch, stale = val_rmse, epoch, 0
            torch.save({"state_dict": model.state_dict(), "input_dim": prepared.x_train.shape[1],
                        "hidden_dims": list(params["hidden_dims"]), "dropout": params["dropout"],
                        "config": params, "best_epoch": best_epoch, "best_val_rmse": best_rmse,
                        "preprocessor_path": str(preprocessor_path.relative_to(root_path(".")))}, checkpoint_path)
        else:
            stale += 1
            if stale >= int(config["early_stopping_patience"]):
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["state_dict"])
    actual_rmse = rmse_log(prepared.y_val, predict_loader(model, val_loader, device))
    if not np.isclose(actual_rmse, best_rmse, atol=1e-6):
        raise RuntimeError("Reloaded checkpoint does not reproduce best validation RMSE")
    history_frame = pd.DataFrame(history)
    metrics_dir = root_path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    result_name = result_name or name.lower()
    history_frame.to_csv(metrics_dir / f"{result_name}_history.csv", index=False)
    save_figures(history_frame, prepared.y_val, predict_loader(model, val_loader, device), result_name)
    row = {"experiment_name": name, "architecture": "-".join(map(str, params["hidden_dims"])),
           "dropout": str(params["dropout"]), "learning_rate": float(params["learning_rate"]),
           "weight_decay": float(params["weight_decay"]), "best_epoch": best_epoch,
           "best_val_rmse": best_rmse}
    if record_experiment:
        _save_row(row, metrics_dir / "experiments.csv")
    print(f"{name} best epoch {best_epoch} | Val RMSE {best_rmse:.6f}", flush=True)
    return row


def run_baseline() -> dict:
    config = load_config()
    train, test = read_data()
    prepared = prepare_data(split_data(train, test, int(config["seed"]), float(config.get("test_size", 0.2))))
    result = train_experiment("E1", prepared, config)
    shutil.copy2(root_path("models/experiments/E1/best_mlp.pt"), root_path("models/best_mlp.pt"))
    return result


def run_all_experiments() -> pd.DataFrame:
    config = load_config()
    train, test = read_data()
    prepared = prepare_data(split_data(train, test, int(config["seed"]), float(config.get("test_size", 0.2))))
    for name in sorted(config["experiments"]):
        train_experiment(name, prepared, config)
    return pd.read_csv(root_path("outputs/metrics/experiments.csv"))


def run_final() -> dict:
    config = load_config()
    table = pd.read_csv(root_path("outputs/metrics/experiments.csv"))
    expected = set(config["experiments"])
    if set(table.experiment_name) != expected:
        raise ValueError("Run all E0-E4 experiments before final training")
    winner = table.sort_values(["best_val_rmse", "experiment_name"]).iloc[0].experiment_name
    train, test = read_data()
    prepared = prepare_data(split_data(train, test, int(config["seed"]), float(config.get("test_size", 0.2))))
    result = train_experiment(winner, prepared, config, artifact_dir="models/final",
                              result_name="final", record_experiment=False)
    (root_path("models/final") / "selected_experiment.txt").write_text(winner, encoding="utf-8")
    return result
