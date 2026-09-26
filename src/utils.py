"""Paths, reproducibility, and device selection."""
from pathlib import Path
import random

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def root_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> tuple[torch.device, str]:
    if requested == "cpu":
        return torch.device("cpu"), "CPU requested"
    if requested not in ("auto", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cpu"), "CUDA unavailable in installed PyTorch"
    try:
        x = torch.randn(2, 2, device="cuda", requires_grad=True)
        x.square().sum().backward()
        torch.cuda.synchronize()
        return torch.device("cuda"), torch.cuda.get_device_name(0)
    except Exception as exc:
        if requested == "cuda":
            raise RuntimeError("CUDA forward/backward probe failed") from exc
        return torch.device("cpu"), f"CUDA probe failed: {exc}"
