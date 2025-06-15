"""
Five coarse-grained actions. Return JSON-serialisable dicts.
粗粒度操作定義：列舉 5 種 macro-level 修改
"""
import random
from typing import List, Dict, Any

COARSE = [
    {"type": "add_polar_group", "params": {"group": "OH", "position": "random"}},
    {"type": "add_polar_group", "params": {"group": "NH2", "position": "random"}},
    {"type": "add_polar_group", "params": {"group": "COOH", "position": "random"}},
    {"type": "increase_pi_system", "params": {"rings": 1, "type": "benzene"}},
    {"type": "increase_pi_system", "params": {"rings": 1, "type": "pyridine"}},
    {"type": "decrease_molecular_weight", "params": {"remove_heavy": True, "target_atoms": ["Cl", "Br", "I"]}},
    {"type": "decrease_molecular_weight", "params": {"remove_methyl": True}},
    {"type": "swap_heteroatom", "params": {"from": "O", "to": "N"}},
    {"type": "swap_heteroatom", "params": {"from": "N", "to": "O"}},
    {"type": "swap_heteroatom", "params": {"from": "S", "to": "O"}},
    {"type": "cyclize", "params": {"size": 5, "type": "saturated"}},
    {"type": "cyclize", "params": {"size": 6, "type": "aromatic"}},
]


def sample(parent_smiles: str, k: int) -> List[Dict[str, Any]]:
    """從粗操作集中隨機選 k 種操作"""
    return random.sample(COARSE, k=min(k, len(COARSE)))