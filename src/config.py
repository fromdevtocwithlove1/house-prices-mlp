"""Read and validate the YAML experiment configuration."""
from pathlib import Path

import yaml

from .utils import root_path


def load_config(path: str | Path = "configs/baseline.yaml") -> dict:
    with root_path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    required = {"seed", "batch_size", "max_epochs", "early_stopping_patience", "experiments"}
    if not required.issubset(config) or not config["experiments"]:
        raise ValueError("Incomplete training config")
    return config
