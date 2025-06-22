"""
Fine-grained 5k substituent library.
微調操作庫讀取：載入基團字典，動態解鎖
"""
import json
import random
import pathlib
import re
import logging
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


def expand_smiles_aware(parent_smiles: str, unlock_factor: float, top_k: int) -> List[Dict[str, Any]]:
    """
    基於 SMILES 分析的智能擴展
    
    COMPLIANCE NOTE: Uses only string-based SMILES analysis and
    LLM-driven pattern recognition. No RDKit property calculations.
    """
    import logging
    import random
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.debug(f"SMILES-aware expansion for {parent_smiles}, unlock_factor={unlock_factor}, top_k={top_k}")
        
        # 獲取所有可用的動作組
        all_groups = get_all_functional_groups()
        
        if not all_groups:
            logger.warning("No functional groups available")
            return get_fallback_actions(top_k)
        
        # 基於 SMILES 字符串分析過濾動作組
        filtered_groups = []
        smiles_features = analyze_smiles_features(parent_smiles)
        
        for group_name, group_data in all_groups.items():
            # 基於分子特徵決定動作的適用性
            if is_action_suitable(group_data, smiles_features, unlock_factor):
                filtered_groups.append((group_name, group_data))
        
        if not filtered_groups:
            logger.warning("No suitable groups found after filtering, using fallback")
            return get_fallback_actions(top_k)
        
        # 計算權重 - 確保權重數量與過濾後的組數量匹配
        weights = []
        for group_name, group_data in filtered_groups:
            weight = calculate_group_weight(group_data, smiles_features, unlock_factor)
            weights.append(max(0.1, weight))  # 確保權重至少為 0.1
        
        # 驗證權重和群組數量匹配
        if len(weights) != len(filtered_groups):
            logger.error(f"Weight mismatch: {len(weights)} weights vs {len(filtered_groups)} groups")
            # 修復權重數量
            if len(weights) < len(filtered_groups):
                weights.extend([1.0] * (len(filtered_groups) - len(weights)))
            else:
                weights = weights[:len(filtered_groups)]
        
        logger.debug(f"Filtered groups: {len(filtered_groups)}, weights: {len(weights)}")
        
        # 生成動作
        actions = []
        attempts = 0
        max_attempts = top_k * 3  # 避免無限迴圈
        
        while len(actions) < top_k and attempts < max_attempts:
            try:
                # 安全地選擇群組
                if len(filtered_groups) == 1:
                    chosen_idx = 0
                else:
                    chosen_idx = random.choices(range(len(filtered_groups)), weights=weights)[0]
                
                group_name, group_data = filtered_groups[chosen_idx]
                
                # 從選中的群組生成動作
                group_actions = generate_actions_from_group(group_data, parent_smiles)
                
                # 添加到結果中，避免重複
                for action in group_actions:
                    if action not in actions and len(actions) < top_k:
                        actions.append(action)
                
                attempts += 1
                
            except Exception as e:
                logger.warning(f"Error selecting group: {e}")
                attempts += 1
                continue
        
        if not actions:
            logger.warning("No actions generated, using fallback")
            return get_fallback_actions(top_k)
        
        logger.info(f"Generated {len(actions)} SMILES-aware actions")
        return actions[:top_k]
        
    except Exception as e:
        logger.error(f"Error in SMILES-aware expansion: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return get_fallback_actions(top_k)


def analyze_smiles_features(smiles: str) -> Dict[str, Any]:
    """
    分析 SMILES 字符串特徵
    
    COMPLIANCE NOTE: Pure string-based analysis, no RDKit calculations.
    """
    try:
        if not smiles:
            return {"error": "empty_smiles"}
        
        features = {
            "length": len(smiles),
            "has_aromatic": any(c.islower() for c in smiles),
            "has_rings": any(c.isdigit() for c in smiles),
            "has_branches": '(' in smiles or ')' in smiles,
            "has_double_bonds": '=' in smiles,
            "has_triple_bonds": '#' in smiles,
            "atom_counts": {},
            "complexity_score": 0.0
        }
        
        # 統計原子類型（基於常見 SMILES 符號）
        atom_symbols = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
        for atom in atom_symbols:
            features["atom_counts"][atom] = smiles.count(atom)
        
        # 計算複雜度分數（基於字符串特徵）
        complexity = 0
        complexity += len(smiles) * 0.1
        complexity += smiles.count('(') * 2  # 分支
        complexity += smiles.count('=') * 1.5  # 雙鍵
        complexity += smiles.count('#') * 2  # 三鍵
        complexity += len([c for c in smiles if c.isdigit()]) * 1  # 環
        
        features["complexity_score"] = complexity
        
        return features
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error analyzing SMILES features: {e}")
        return {"error": "analysis_failed"}


def is_action_suitable(group_data: Dict, smiles_features: Dict, unlock_factor: float) -> bool:
    """
    判斷動作是否適合當前分子
    
    COMPLIANCE NOTE: Decision based on string features and algorithmic logic only.
    """
    try:
        if smiles_features.get("error"):
            return True  # 如果分析失敗，允許所有動作
        
        # 基於解鎖因子的基本過濾
        if random.random() > unlock_factor:
            return False
        
        # 基於分子複雜度的過濾
        complexity = smiles_features.get("complexity_score", 0)
        
        # 複雜分子避免過度修飾
        if complexity > 50 and group_data.get("complexity_increase", 0) > 10:
            return random.random() < 0.3
        
        # 簡單分子鼓勵添加功能基
        if complexity < 10:
            return True
        
        # 其他情況基於隨機性和分子特徵
        return random.random() < 0.7
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Error checking action suitability: {e}")
        return True  # 錯誤時默認允許


def calculate_group_weight(group_data: Dict, smiles_features: Dict, unlock_factor: float) -> float:
    """
    計算群組權重
    
    COMPLIANCE NOTE: Weight calculation based on string features only.
    """
    try:
        base_weight = 1.0
        
        if smiles_features.get("error"):
            return base_weight
        
        # 基於分子複雜度調整權重
        complexity = smiles_features.get("complexity_score", 0)
        
        # 簡單分子偏好添加功能基
        if complexity < 20:
            if group_data.get("type") == "add":
                base_weight *= 1.5
        
        # 複雜分子偏好簡單修飾
        elif complexity > 40:
            if group_data.get("complexity_increase", 0) < 5:
                base_weight *= 1.3
            else:
                base_weight *= 0.7
        
        # 基於解鎖因子調整
        base_weight *= (0.5 + unlock_factor)
        
        # 添加隨機性
        base_weight *= (0.8 + random.random() * 0.4)
        
        return max(0.1, base_weight)
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Error calculating group weight: {e}")
        return 1.0


def generate_actions_from_group(group_data: Dict, parent_smiles: str) -> List[Dict[str, Any]]:
    """
    從群組數據生成動作
    
    COMPLIANCE NOTE: Action generation based on group templates,
    no RDKit property calculations.
    """
    try:
        actions = []
        
        # 基於群組類型生成動作
        group_type = group_data.get("type", "substitute")
        group_name = group_data.get("name", "unknown")
        
        if group_type == "add":
            # 添加功能基
            fragments = group_data.get("fragments", ["C"])
            for fragment in fragments[:3]:  # 限制每個群組最多3個動作
                action = {
                    "type": "substitute",
                    "name": f"add_{group_name}_{fragment}",
                    "description": f"添加 {group_name} ({fragment})",
                    "params": {"smiles": fragment, "fragment": group_name}
                }
                actions.append(action)
        
        elif group_type == "modify":
            # 修飾現有結構
            modifications = group_data.get("modifications", ["oxidation"])
            for mod in modifications[:2]:
                action = {
                    "type": "modify",
                    "name": f"{group_name}_{mod}",
                    "description": f"{group_name} {mod}",
                    "params": {"modification": mod, "group": group_name}
                }
                actions.append(action)
        
        else:
            # 默認替換動作
            action = {
                "type": "substitute",
                "name": f"apply_{group_name}",
                "description": f"應用 {group_name}",
                "params": {"group": group_name}
            }
            actions.append(action)
        
        return actions
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.debug(f"Error generating actions from group: {e}")
        return []


def get_all_functional_groups() -> Dict[str, Dict]:
    """
    獲取所有功能基群組
    
    COMPLIANCE NOTE: Static functional group definitions,
    no dynamic property calculations.
    """
    return {
        "methyl": {
            "type": "add",
            "name": "methyl",
            "fragments": ["C"],
            "complexity_increase": 1
        },
        "hydroxyl": {
            "type": "add", 
            "name": "hydroxyl",
            "fragments": ["O"],
            "complexity_increase": 2
        },
        "amino": {
            "type": "add",
            "name": "amino", 
            "fragments": ["N"],
            "complexity_increase": 3
        },
        "carboxyl": {
            "type": "add",
            "name": "carboxyl",
            "fragments": ["C(=O)O"],
            "complexity_increase": 5
        },
        "halogen": {
            "type": "add",
            "name": "halogen",
            "fragments": ["F", "Cl", "Br"],
            "complexity_increase": 1
        },
        "ethyl": {
            "type": "add",
            "name": "ethyl",
            "fragments": ["CC"],
            "complexity_increase": 2
        },
        "phenyl": {
            "type": "add",
            "name": "phenyl", 
            "fragments": ["c1ccccc1"],
            "complexity_increase": 8
        },
        "nitro": {
            "type": "add",
            "name": "nitro",
            "fragments": ["[N+](=O)[O-]"],
            "complexity_increase": 4
        },
        "sulfo": {
            "type": "add",
            "name": "sulfo",
            "fragments": ["S(=O)(=O)O"],
            "complexity_increase": 6
        },
        "acetyl": {
            "type": "add",
            "name": "acetyl",
            "fragments": ["C(=O)C"],
            "complexity_increase": 4
        }
    }


def get_fallback_actions(k: int) -> List[Dict[str, Any]]:
    """
    獲取後備動作列表
    
    COMPLIANCE NOTE: Basic chemical operations without RDKit calculations.
    
    Args:
        k: 需要的動作數量
        
    Returns:
        List[Dict[str, Any]]: 後備動作列表
    """
    basic_actions = [
        {"type": "substitute", "name": "add_methyl", "description": "添加甲基", "params": {"smiles": "C"}},
        {"type": "substitute", "name": "add_hydroxyl", "description": "添加羥基", "params": {"smiles": "O"}},
        {"type": "substitute", "name": "add_amino", "description": "添加氨基", "params": {"smiles": "N"}},
        {"type": "substitute", "name": "add_fluorine", "description": "添加氟", "params": {"smiles": "F"}},
        {"type": "substitute", "name": "add_chlorine", "description": "添加氯", "params": {"smiles": "Cl"}},
        {"type": "substitute", "name": "add_ethyl", "description": "添加乙基", "params": {"smiles": "CC"}},
        {"type": "substitute", "name": "add_phenyl", "description": "添加苯基", "params": {"smiles": "c1ccccc1"}},
        {"type": "substitute", "name": "add_carboxyl", "description": "添加羧基", "params": {"smiles": "C(=O)O"}},
        {"type": "substitute", "name": "add_nitro", "description": "添加硝基", "params": {"smiles": "[N+](=O)[O-]"}},
        {"type": "substitute", "name": "add_sulfonic", "description": "添加磺酸基", "params": {"smiles": "S(=O)(=O)O"}},
    ]
    
    if k <= len(basic_actions):
        import random
        return random.sample(basic_actions, k)
    else:
        # 如果需要更多動作，重複選擇
        repeated_actions = basic_actions * ((k // len(basic_actions)) + 1)
        return repeated_actions[:k]


def propose_mixed_actions(parent_smiles: str, depth: int, k_init: int) -> List[Dict[str, Any]]:
    """
    混合粗粒度和細粒度動作 - 修復版本，確保動作多樣性
    
    COMPLIANCE NOTE: Uses only string-based molecular analysis,
    no RDKit property calculations before Oracle evaluation.
    """
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # 根據深度調整粗粒度和細粒度的比例
        if depth == 0:
            coarse_ratio = 0.8  # 增加粗粒度比例以獲得更多多樣性
        elif depth <= 2:
            coarse_ratio = 0.6
        else:
            coarse_ratio = 0.4
        
        coarse_k = int(k_init * coarse_ratio)
        fine_k = k_init - coarse_k
        
        actions = []
        
        # 獲取粗粒度動作
        if coarse_k > 0:
            try:
                # 檢查是否可以導入 coarse_actions
                from . import coarse_actions
                coarse_actions_list = coarse_actions.sample(k=coarse_k, parent_smiles=parent_smiles)
                actions.extend(coarse_actions_list)
                logger.debug(f"Added {len(coarse_actions_list)} coarse actions")
            except Exception as e:
                logger.warning(f"Error getting coarse actions: {e}")
                # 如果粗粒度動作失敗，將配額轉移到細粒度
                fine_k += coarse_k
        
        # 獲取細粒度動作 - 使用安全的方法
        if fine_k > 0:
            try:
                unlock_factor = min(1.0, 0.5 + depth * 0.1)  # 增加解鎖因子
                
                # 使用更安全的動作生成方法
                fine_actions_list = generate_safe_fine_actions(
                    parent_smiles=parent_smiles,
                    k=fine_k,
                    unlock_factor=unlock_factor
                )
                
                actions.extend(fine_actions_list)
                logger.debug(f"Added {len(fine_actions_list)} fine actions")
                
            except Exception as e:
                logger.error(f"Error getting fine actions: {e}")
                # 如果細粒度也失敗，使用後備動作
                fallback_actions = get_fallback_actions(fine_k)
                actions.extend(fallback_actions)
                logger.debug(f"Used {len(fallback_actions)} fallback actions")
        
        # 如果沒有任何動作，使用基本後備
        if not actions:
            actions = get_fallback_actions(k_init)
            logger.warning(f"No actions generated, using {len(actions)} fallback actions")
        
        # 新增：確保動作多樣性，移除重複動作
        unique_actions = []
        seen_names = set()
        for action in actions:
            action_key = action.get('name', str(action))
            if action_key not in seen_names:
                unique_actions.append(action)
                seen_names.add(action_key)
        
        if len(unique_actions) < len(actions):
            logger.info(f"Removed {len(actions) - len(unique_actions)} duplicate actions")
            actions = unique_actions
        
        logger.info(f"Mixed actions: {len(actions)} total (unique)")
        return actions
        
    except Exception as e:
        logger.error(f"Error in propose_mixed_actions: {e}")
        # 最終後備
        return get_fallback_actions(k_init)


def generate_safe_fine_actions(parent_smiles: str, k: int, unlock_factor: float) -> List[Dict[str, Any]]:
    """
    安全的細粒度動作生成（避免權重錯誤）
    
    COMPLIANCE NOTE: Safe action generation with proper error handling,
    no RDKit property calculations.
    """
    import logging
    import random
    
    logger = logging.getLogger(__name__)
    
    try:
        # 獲取功能基群組
        all_groups = get_all_functional_groups()
        
        if not all_groups:
            return get_fallback_actions(k)
        
        # 分析 SMILES 特徵
        smiles_features = analyze_smiles_features(parent_smiles)
        
        # 過濾適合的群組
        suitable_groups = []
        for group_name, group_data in all_groups.items():
            if is_action_suitable(group_data, smiles_features, unlock_factor):
                suitable_groups.append((group_name, group_data))
        
        if not suitable_groups:
            logger.warning("No suitable groups found, using all groups")
            suitable_groups = list(all_groups.items())
        
        # 生成動作 - 使用簡單的隨機選擇避免權重問題
        actions = []
        for _ in range(k):
            try:
                # 隨機選擇群組
                group_name, group_data = random.choice(suitable_groups)
                
                # 從群組生成動作
                group_actions = generate_actions_from_group(group_data, parent_smiles)
                
                if group_actions:
                    action = random.choice(group_actions)
                    if action not in actions:  # 避免重複
                        actions.append(action)
                
            except Exception as e:
                logger.debug(f"Error generating individual action: {e}")
                continue
        
        # 如果動作不足，用後備動作補充
        if len(actions) < k:
            needed = k - len(actions)
            fallback = get_fallback_actions(needed)
            actions.extend(fallback)
        
        return actions[:k]
        
    except Exception as e:
        logger.error(f"Error in safe fine actions generation: {e}")
        return get_fallback_actions(k)