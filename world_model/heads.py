"""Multiple prediction heads for the world model."""
from typing import Dict
import torch
from torch import nn


class PropertyHeads(nn.Module):
    """Placeholder heads predicting synthesis, toxicity etc."""

    def __init__(self, in_dim: int, dropout: float = 0.1):
        super().__init__()
        self.synth = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        self.tox = nn.Sequential(nn.Linear(in_dim, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.dropout(x)
        return {
            "p_synth": self.synth(x).squeeze(-1),
            "p_tox": self.tox(x).squeeze(-1),
            "sigma_sum": x.std().unsqueeze(0),
            "cost_norm": torch.sigmoid(x.mean()).unsqueeze(0),
        }
