"""
Five coarse-grained actions.  Return JSON-serialisable dicts.
"""
import random

COARSE = [
    {"type": "add_polar",     "params": {"group": "OH"}},
    {"type": "increase_pi",   "params": {"rings": 1}},
    {"type": "decrease_mw",   "params": {"remove_heavy": True}},
    {"type": "swap_hetero",   "params": {"from": "O", "to": "N"}},
    {"type": "cyclize",       "params": {"size": 5}},
]


def sample(parent_smiles: str, k: int):
    return random.sample(COARSE, k=min(k, len(COARSE)))