"""
Fine-grained 5k substituent library.
微調操作庫讀取：載入基團字典，動態解鎖
"""
import json
import random
import pathlib
import re
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


def analyze_smiles_heuristics(smiles: str) -> Dict[str, Any]:
    """
    基於 SMILES 字符串的啟發式分析，不依賴 RDKit
    
    Args:
        smiles: SMILES 字符串
    
    Returns:
        分子特性的啟發式估計
    """
    if not smiles:
        return {"length": 0, "polar_atoms": 0, "aromatic_rings": 0, "heavy_atoms": 0}
    
    # 計算基本特徵
    length = len(smiles)
    
    # 極性原子計數（O, N, S, P）
    polar_atoms = smiles.count('O') + smiles.count('N') + smiles.count('S') + smiles.count('P')
    
    # 芳香環估計（基於芳香性標記 'c', 'n', 'o', 's'）
    aromatic_markers = smiles.count('c') + smiles.count('n') + smiles.count('o') + smiles.count('s')
    aromatic_rings = max(aromatic_markers // 6, 0)  # 粗略估計
    
    # 重原子計數（排除氫）
    heavy_atoms = len([c for c in smiles if c.isupper() and c not in ['H']])
    
    # 鹵素原子
    halogens = smiles.count('Cl') + smiles.count('Br') + smiles.count('F') + smiles.count('I')
    
    # 環結構計數（基於數字標記）
    ring_numbers = len(re.findall(r'\d', smiles))
    
    return {
        "length": length,
        "polar_atoms": polar_atoms,
        "aromatic_rings": aromatic_rings,
        "heavy_atoms": heavy_atoms,
        "halogens": halogens,
        "ring_numbers": ring_numbers,
        "complexity_score": length + polar_atoms * 2 + aromatic_rings * 3
    }


def expand_smiles_aware(parent_smiles: str, k: int = 5) -> List[Dict]:
    """
    根據父分子特性選擇合適的 fine actions
    
    Args:
        parent_smiles: 父分子的 SMILES 字符串
        k: 要選擇的動作數量
    
    Returns:
        選擇的動作列表
    """
    all_actions = get_all_actions()
    
    if k >= len(all_actions):
        return all_actions
    
    # 簡單的啟發式規則：根據分子長度和複雜度選擇動作
    mol_length = len(parent_smiles)
    
    # 為不同類型的動作分配權重
    weights = []
    for action in all_actions:
        action_type = action.get('type', 'unknown')
        
        if mol_length < 20:  # 小分子，偏好添加基團
            if action_type in ['alkyl_group', 'functional_group']:
                weights.append(2.0)
            else:
                weights.append(1.0)
        else:  # 大分子，偏好移除或替換
            if action_type in ['removal', 'replacement']:
                weights.append(2.0)
            else:
                weights.append(1.0)
    
    # 加權隨機選擇
    import random
    selected_actions = random.choices(all_actions, weights=weights, k=k)
    
    return selected_actions


def expand_smiles_aware(parent_smiles: str, unlock_factor: float, top_k: int = 30) -> List[Dict[str, Any]]:
    """
    基於 SMILES 啟發式分析的微調操作擴展，不依賴 RDKit
    
    Args:
        parent_smiles: 父分子 SMILES
        unlock_factor: 解鎖因子，基於訪問次數
        top_k: 最多返回的操作數量
    
    Returns:
        針對當前分子特性優化的微調操作列表
    """
    if not _GROUPS:
        return []
    
    # SMILES 啟發式分析
    props = analyze_smiles_heuristics(parent_smiles)
    
    # 根據啟發式特性智能過濾和評分操作
    filtered_groups = []
    
    for group in _GROUPS:
        score = 1.0  # 基礎分數
        group_smiles = group["params"]["smiles"]
        group_props = analyze_smiles_heuristics(group_smiles)
        
        # 根據分子複雜度調整策略
        if props["complexity_score"] > 50:  # 複雜分子，優先小基團
            if group_props["length"] <= 3:
                score *= 1.5
        elif props["complexity_score"] < 20:  # 簡單分子，可以添加較大基團
            if group_props["length"] > 3:
                score *= 1.3
        
        # 根據極性原子數量調整策略
        if props["polar_atoms"] < 2:  # 極性不足，優先極性基團
            if group_props["polar_atoms"] > 0:
                score *= 1.4
        elif props["polar_atoms"] > 5:  # 極性過高，優先非極性基團
            if group_props["polar_atoms"] == 0:
                score *= 1.3
        
        # 根據芳香環調整
        if props["aromatic_rings"] == 0:  # 無芳香環，可考慮添加
            if group["type"] == "add_ring":
                score *= 1.2
        elif props["aromatic_rings"] > 2:  # 芳香環過多，避免再添加
            if group["type"] == "add_ring":
                score *= 0.5
        
        # 根據鹵素原子調整
        if props["halogens"] > 2:  # 鹵素過多，避免再添加
            if any(hal in group_smiles for hal in ['Cl', 'Br', 'F', 'I']):
                score *= 0.7
        
        filtered_groups.append((group, score))
    
    # 按分數排序
    filtered_groups.sort(key=lambda x: x[1], reverse=True)
    
    # 根據 unlock_factor 和評分選擇操作
    available_actions = min(int(unlock_factor * len(filtered_groups)), len(filtered_groups))
    k = min(available_actions, top_k)
    
    if k <= 0:
        return []
    
    # 加權隨機選擇，高分數的操作有更高機率被選中
    selected_groups = []
    weights = [score for _, score in filtered_groups[:available_actions]]
    
    for _ in range(k):
        if not filtered_groups:
            break
        
        # 加權隨機選擇
        chosen_idx = random.choices(range(len(filtered_groups)), weights=weights)[0]
        selected_groups.append(filtered_groups[chosen_idx][0])
        
        # 移除已選擇的操作避免重複
        filtered_groups.pop(chosen_idx)
        weights.pop(chosen_idx)
    
    return selected_groups


def expand_llm_guided(parent_smiles: str, llm_feedback: Dict[str, str], 
                     unlock_factor: float, top_k: int = 30) -> List[Dict[str, Any]]:
    """
    基於 LLM 回饋的微調操作選擇
    
    Args:
        parent_smiles: 父分子 SMILES
        llm_feedback: LLM 對當前分子的分析回饋，包含建議的修改方向
        unlock_factor: 解鎖因子
        top_k: 最多返回的操作數量
    
    Returns:
        基於 LLM 建議的微調操作列表
    """
    if not _GROUPS or not llm_feedback:
        return expand_smiles_aware(parent_smiles, unlock_factor, top_k)
    
    scored_groups = []
    
    # 從 LLM 回饋中提取關鍵字
    feedback_text = " ".join(llm_feedback.values()).lower()
    
    for group in _GROUPS:
        score = 1.0
        group_smiles = group["params"]["smiles"]
        group_name = group["params"].get("fragment", "")
        
        # 根據 LLM 回饋調整分數
        if "increase" in feedback_text and "polarity" in feedback_text:
            if any(polar in group_smiles for polar in ['O', 'N', 'S']):
                score *= 1.5
        
        if "decrease" in feedback_text and "size" in feedback_text:
            if len(group_smiles) <= 2:
                score *= 1.4
        
        if "add" in feedback_text and "ring" in feedback_text:
            if group["type"] == "add_ring":
                score *= 1.6
        
        if "hydrophobic" in feedback_text:
            if group_smiles.count('C') > group_smiles.count('O') + group_smiles.count('N'):
                score *= 1.3
        
        if "hydrophilic" in feedback_text:
            if group_smiles.count('O') + group_smiles.count('N') > 0:
                score *= 1.3
        
        # 特定基團建議
        if "methyl" in feedback_text and "methyl" in group_name.lower():
            score *= 2.0
        if "hydroxyl" in feedback_text and "hydroxyl" in group_name.lower():
            score *= 2.0
        if "amino" in feedback_text and "amino" in group_name.lower():
            score *= 2.0
        
        scored_groups.append((group, score))
    
    # 排序並選擇
    scored_groups.sort(key=lambda x: x[1], reverse=True)
    available_actions = min(int(unlock_factor * len(scored_groups)), len(scored_groups))
    k = min(available_actions, top_k)
    
    return [group for group, _ in scored_groups[:k]]


def get_action_statistics() -> Dict[str, Any]:
    """返回操作庫的統計資訊"""
    if not _GROUPS:
        return {"total": 0, "types": {}}
    
    type_counts = {}
    for group in _GROUPS:
        action_type = group["type"]
        type_counts[action_type] = type_counts.get(action_type, 0) + 1
    
    return {
        "total": len(_GROUPS),
        "types": type_counts,
        "substitute_ratio": type_counts.get("substitute", 0) / len(_GROUPS),
        "add_ring_ratio": type_counts.get("add_ring", 0) / len(_GROUPS)
    }


def expand_balanced(parent_smiles: str, unlock_factor: float, top_k: int = 30, 
                   exploration_weight: float = 0.3) -> List[Dict[str, Any]]:
    """
    平衡探索與利用的微調操作選擇
    
    Args:
        parent_smiles: 父分子 SMILES
        unlock_factor: 解鎖因子
        top_k: 最多返回的操作數量
        exploration_weight: 探索權重 (0-1)，較高值增加隨機性
    
    Returns:
        平衡探索與利用的微調操作列表
    """
    if not _GROUPS:
        return []
    
    # 混合智能選擇和隨機選擇
    smiles_aware_k = int(top_k * (1 - exploration_weight))
    random_k = top_k - smiles_aware_k
    
    results = []
    
    # 獲取基於 SMILES 分析的智能選擇操作
    if smiles_aware_k > 0:
        smart_actions = expand_smiles_aware(parent_smiles, unlock_factor, smiles_aware_k)
        results.extend(smart_actions)
    
    # 獲取隨機選擇的操作
    if random_k > 0:
        remaining_groups = [g for g in _GROUPS if g not in results]
        if remaining_groups:
            available_random = min(int(unlock_factor * len(remaining_groups)), len(remaining_groups))
            k_random = min(random_k, available_random)
            if k_random > 0:
                random_actions = random.sample(remaining_groups, k=k_random)
                results.extend(random_actions)
    
    return results


def get_all_actions():
    """
    返回所有可用的 fine actions 列表
    """
    actions = []
    
    # 小分子基團
    small_groups = [
        {"name": "add_methyl", "description": "添加甲基 (-CH3)", "type": "alkyl_group"},
        {"name": "add_ethyl", "description": "添加乙基 (-C2H5)", "type": "alkyl_group"},
        {"name": "add_propyl", "description": "添加丙基 (-C3H7)", "type": "alkyl_group"},
        {"name": "add_isopropyl", "description": "添加異丙基", "type": "alkyl_group"},
        {"name": "add_butyl", "description": "添加丁基 (-C4H9)", "type": "alkyl_group"},
        {"name": "add_tert_butyl", "description": "添加叔丁基", "type": "alkyl_group"},
    ]
    
    # 官能基
    functional_groups = [
        {"name": "add_hydroxyl", "description": "添加羥基 (-OH)", "type": "functional_group"},
        {"name": "add_amino", "description": "添加氨基 (-NH2)", "type": "functional_group"},
        {"name": "add_carboxyl", "description": "添加羧基 (-COOH)", "type": "functional_group"},
        {"name": "add_carbonyl", "description": "添加羰基 (=O)", "type": "functional_group"},
        {"name": "add_ester", "description": "添加酯基 (-COO-)", "type": "functional_group"},
        {"name": "add_ether", "description": "添加醚鍵 (-O-)", "type": "functional_group"},
        {"name": "add_amide", "description": "添加醯胺基 (-CONH-)", "type": "functional_group"},
        {"name": "add_nitrile", "description": "添加腈基 (-CN)", "type": "functional_group"},
        {"name": "add_nitro", "description": "添加硝基 (-NO2)", "type": "functional_group"},
        {"name": "add_sulfone", "description": "添加磺醯基 (-SO2-)", "type": "functional_group"},
    ]
    
    # 鹵素
    halogens = [
        {"name": "add_fluorine", "description": "添加氟原子 (-F)", "type": "halogen"},
        {"name": "add_chlorine", "description": "添加氯原子 (-Cl)", "type": "halogen"},
        {"name": "add_bromine", "description": "添加溴原子 (-Br)", "type": "halogen"},
        {"name": "add_iodine", "description": "添加碘原子 (-I)", "type": "halogen"},
    ]
    
    # 結構修飾
    modifications = [
        {"name": "remove_group", "description": "移除官能基", "type": "removal"},
        {"name": "replace_group", "description": "替換官能基", "type": "replacement"},
        {"name": "position_change", "description": "改變取代位置", "type": "positional"},
        {"name": "stereochemistry", "description": "改變立體化學", "type": "stereochemical"},
        {"name": "double_bond", "description": "引入雙鍵", "type": "unsaturation"},
        {"name": "triple_bond", "description": "引入三鍵", "type": "unsaturation"},
    ]
    
    # 合併所有 actions
    actions.extend(small_groups)
    actions.extend(functional_groups)
    actions.extend(halogens)
    actions.extend(modifications)
    
    return actions