"""Configurable fully connected regressor for log-price."""
from collections.abc import Sequence

import torch
from torch import nn


class HousePriceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64, 32),
                 dropout: float | Sequence[float] = (0.15, 0.10)):
        super().__init__()
        if input_dim < 1 or not hidden_dims or any(int(d) < 1 for d in hidden_dims):
            raise ValueError("All layer dimensions must be positive")
        count = max(0, len(hidden_dims) - 1)
        rates = [float(dropout)] * count if isinstance(dropout, (int, float)) else list(dropout)
        if len(rates) != count or any(not 0 <= rate < 1 for rate in rates):
            raise ValueError("Dropout must specify each non-final hidden layer")
        layers = []
        previous = input_dim
        for index, width in enumerate(hidden_dims):
            layers.append(nn.Linear(previous, width))
            if index < count:
                layers.extend([nn.BatchNorm1d(width), nn.ReLU()])
                if rates[index] > 0:
                    layers.append(nn.Dropout(rates[index]))
            else:
                layers.append(nn.ReLU())
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
