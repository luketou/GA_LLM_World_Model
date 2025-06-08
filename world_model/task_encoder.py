"""Task encoder supporting GuacaMol benchmark suite."""
from typing import List
import torch
from torch import nn

from guacamol.benchmark_suites import goal_directed_benchmark_suite

TASK_LIST = [b.name for b in goal_directed_benchmark_suite()]


class TaskEncoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(len(TASK_LIST), dim)

    def forward(self, task_name: str) -> torch.Tensor:
        idx = TASK_LIST.index(task_name)
        idx_tensor = torch.tensor(idx, dtype=torch.long)
        return self.embed(idx_tensor)
