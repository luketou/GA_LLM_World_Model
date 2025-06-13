"""
Fine-grained 5k substituent library.
"""
import json, random, pathlib
_GROUPS = json.loads(
    (pathlib.Path(__file__).with_suffix("_groups.json")).read_text()
)  # TODO:← 請準備此 JSON

def expand(parent: str, unlock_factor: float, top_k: int = 30):
    k = min(int(unlock_factor), top_k, len(_GROUPS))
    return random.sample(_GROUPS, k=k)