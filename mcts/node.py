from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Node:
    smiles: str
    depth: int
    visits: int = 0
    total_score: float = 0.0
    advantage: float = 0.0
    regret: float = 0.0
    children: Dict[str, "Node"] = field(default_factory=dict)

    @property
    def mean_score(self) -> float:
        return self.total_score / self.visits if self.visits else 0.0