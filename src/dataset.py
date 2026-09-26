"""Torch datasets and deterministic DataLoaders."""
import torch
from torch.utils.data import DataLoader, TensorDataset

from .preprocess import PreparedData


def make_loaders(prepared: PreparedData, batch_size: int, seed: int):
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for BatchNorm")
    x_train = torch.from_numpy(prepared.x_train)
    x_val = torch.from_numpy(prepared.x_val)
    x_test = torch.from_numpy(prepared.x_test)
    y_train = torch.from_numpy(prepared.y_train)
    y_val = torch.from_numpy(prepared.y_val)
    if any(x.dtype != torch.float32 for x in (x_train, x_val, x_test, y_train, y_val)):
        raise TypeError("Tensor dtype must be float32")
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size,
                       shuffle=True, drop_last=(len(x_train) % batch_size == 1), generator=generator)
    val = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test = DataLoader(TensorDataset(x_test), batch_size=batch_size, shuffle=False)
    train_eval = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False)
    return train, val, test, train_eval
