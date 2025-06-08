"""Risk gate implementation."""
from typing import Dict
import torch
from torch import nn


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


class RiskGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(5))

    def forward(self, props: Dict[str, torch.Tensor], novelty: torch.Tensor) -> torch.Tensor:
        p_synth = props["p_synth"]
        p_tox = props["p_tox"]
        sigma_sum = props["sigma_sum"]
        cost_norm = props["cost_norm"]
        x = torch.stack([
            1 - p_synth,
            p_tox,
            sigma_sum,
            cost_norm,
            novelty,
        ])
        risk = sigmoid((self.weights * x).sum())
        return risk
