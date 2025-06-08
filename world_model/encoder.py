"""Stub SE(3)-Transformer style encoder."""
from typing import Any
import torch
from torch import nn


class MoleculeEncoder(nn.Module):
    """Placeholder encoder returning a fixed-size vector."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(2048, dim)

    def forward(self, fp: Any) -> torch.Tensor:
        x = torch.tensor(list(fp), dtype=torch.float32)
        return self.fc(x)
