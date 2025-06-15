"""
Fine-grained 5k substituent library.
微調操作庫讀取：載入基團字典，動態解鎖
"""
import json
import random
import pathlib
from typing import List, Dict, Any

# 載入 5k 基團字典
try:
    _GROUPS = json.loads(
        (pathlib.Path(__file__).parent / "fine_actions_groups.json").read_text()
    )
except FileNotFoundError:
    print("Warning: fine_actions_groups.json not found, using empty list")
    _GROUPS = []


def expand(parent_smiles: str, unlock_factor: float, top_k: int = 30) -> List[Dict[str, Any]]:
    """
    根據父節點訪問次數計算 unlock_factor，取樣 top_k 微調操作
    
    Args:
        parent_smiles: 父分子 SMILES
        unlock_factor: 解鎖因子，通常基於訪問次數
        top_k: 最多返回的操作數量
    
    Returns:
        微調操作列表
    """
    if not _GROUPS:
        return []
    
    # 根據 unlock_factor 計算實際可用的操作數量
    available_actions = min(int(unlock_factor * len(_GROUPS)), len(_GROUPS))
    k = min(available_actions, top_k)
    
    if k <= 0:
        return []
    
    return random.sample(_GROUPS, k=k)