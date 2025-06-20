"""
Comprehensive coarse-grained actions focused on scaffold swapping for diverse exploration.
粗粒度操作定義：以骨架抽換為主的 macro-level 修改，增強化學空間探索多樣性

主要設計思想：
1. 骨架抽換操作（33種）：涵蓋單環、稠環、飽和環及藥物常見骨架
2. 官能基操作（3種）：保留關鍵極性基團添加
3. 結構調整（6種）：包含分子量調節、雜原子交換、成環/開環操作

這種設計能讓 MCTS 在粗粒度層級實現更大幅度的結構跳躍，
避免過度依賴初始分子的骨架特徵，提升全域探索能力。
"""
import random
from typing import List, Dict, Any

COARSE = [
    # 骨架抽換操作 - 單環芳香族系統
    {"type": "scaffold_swap", "params": {"target_scaffold": "benzene", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "pyridine", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "pyrimidine", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "pyrazine", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "pyridazine", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "thiophene", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "furan", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "pyrrole", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "imidazole", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "thiazole", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "oxazole", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "pyrazole", "preserve_substituents": True}},
    
    # 骨架抽換操作 - 稠環芳香族系統
    {"type": "scaffold_swap", "params": {"target_scaffold": "naphthalene", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "quinoline", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "isoquinoline", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "quinazoline", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "quinoxaline", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "benzothiophene", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "benzofuran", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "indole", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "benzimidazole", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "purine", "preserve_substituents": True}},
    
    # 骨架抽換操作 - 飽和環系統
    {"type": "scaffold_swap", "params": {"target_scaffold": "cyclohexane", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "cyclopentane", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "piperidine", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "piperazine", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "morpholine", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "pyrrolidine", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "tetrahydrofuran", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "tetrahydropyran", "preserve_substituents": True}},
    
    # 骨架抽換操作 - 藥物常見骨架
    {"type": "scaffold_swap", "params": {"target_scaffold": "phenyl", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "biphenyl", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "pyrimidinone", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "triazine", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "triazole", "preserve_substituents": True}},
    {"type": "scaffold_swap", "params": {"target_scaffold": "tetrazole", "preserve_substituents": True}},
    
    # 保留少數關鍵的官能基操作
    {"type": "add_polar_group", "params": {"group": "OH", "position": "random"}},
    {"type": "add_polar_group", "params": {"group": "NH2", "position": "random"}},
    {"type": "add_polar_group", "params": {"group": "COOH", "position": "random"}},
    
    # 保留重要的結構調整
    {"type": "decrease_molecular_weight", "params": {"remove_heavy": True, "target_atoms": ["Cl", "Br", "I"]}},
    {"type": "swap_heteroatom", "params": {"from": "O", "to": "N"}},
    {"type": "swap_heteroatom", "params": {"from": "N", "to": "O"}},
    {"type": "cyclize", "params": {"size": 6, "type": "aromatic"}},
    {"type": "ring_opening", "params": {"target_ring_size": 5, "preserve_aromaticity": False}},
    {"type": "ring_opening", "params": {"target_ring_size": 6, "preserve_aromaticity": False}},
]


def sample(k: int = 10, parent_smiles: str = None) -> List[Dict]:
    """
    從 coarse actions 中隨機抽樣 k 個動作
    
    Args:
        k: 要抽樣的動作數量
        parent_smiles: 父分子 SMILES（可選，用於某些特定的過濾邏輯）
    
    Returns:
        抽樣得到的動作列表
    """
    all_actions = get_all_actions()
    
    if k >= len(all_actions):
        return all_actions
    
    # 為 scaffold_swap 類型的動作賦予更高權重
    scaffold_priority_factor = 2.0
    weights = []
    
    for action in all_actions:
        if action.get('type') == 'scaffold_swap':
            weights.append(scaffold_priority_factor)
        else:
            weights.append(1.0)
    
    # 加權隨機抽樣
    import random
    selected_actions = random.choices(all_actions, weights=weights, k=k)
    
    return selected_actions


def get_all_actions():
    """
    返回所有可用的 coarse actions 列表
    """
    actions = []
    
    # 單環芳香族系統 (12種)
    aromatic_single = [
        {"name": "add_benzene", "description": "添加苯環", "type": "scaffold_swap"},
        {"name": "add_pyridine", "description": "添加吡啶環", "type": "scaffold_swap"},
        {"name": "add_pyrimidine", "description": "添加嘧啶環", "type": "scaffold_swap"},
        {"name": "add_pyrazine", "description": "添加吡嗪環", "type": "scaffold_swap"},
        {"name": "add_pyridazine", "description": "添加噠嗪環", "type": "scaffold_swap"},
        {"name": "add_thiophene", "description": "添加噻吩環", "type": "scaffold_swap"},
        {"name": "add_furan", "description": "添加呋喃環", "type": "scaffold_swap"},
        {"name": "add_pyrrole", "description": "添加吡咯環", "type": "scaffold_swap"},
        {"name": "add_imidazole", "description": "添加咪唑環", "type": "scaffold_swap"},
        {"name": "add_thiazole", "description": "添加噻唑環", "type": "scaffold_swap"},
        {"name": "add_oxazole", "description": "添加噁唑環", "type": "scaffold_swap"},
        {"name": "add_pyrazole", "description": "添加吡唑環", "type": "scaffold_swap"},
    ]
    
    # 稠環芳香族系統 (10種)
    aromatic_fused = [
        {"name": "add_naphthalene", "description": "添加萘環", "type": "scaffold_swap"},
        {"name": "add_quinoline", "description": "添加喹啉環", "type": "scaffold_swap"},
        {"name": "add_isoquinoline", "description": "添加異喹啉環", "type": "scaffold_swap"},
        {"name": "add_quinazoline", "description": "添加喹唑啉環", "type": "scaffold_swap"},
        {"name": "add_quinoxaline", "description": "添加喹噁啉環", "type": "scaffold_swap"},
        {"name": "add_benzothiophene", "description": "添加苯並噻吩環", "type": "scaffold_swap"},
        {"name": "add_benzofuran", "description": "添加苯並呋喃環", "type": "scaffold_swap"},
        {"name": "add_indole", "description": "添加吲哚環", "type": "scaffold_swap"},
        {"name": "add_benzimidazole", "description": "添加苯並咪唑環", "type": "scaffold_swap"},
        {"name": "add_purine", "description": "添加嘌呤環", "type": "scaffold_swap"},
    ]
    
    # 飽和環系統 (8種)
    saturated_rings = [
        {"name": "add_cyclohexane", "description": "添加環己烷", "type": "scaffold_swap"},
        {"name": "add_cyclopentane", "description": "添加環戊烷", "type": "scaffold_swap"},
        {"name": "add_piperidine", "description": "添加哌啶環", "type": "scaffold_swap"},
        {"name": "add_piperazine", "description": "添加哌嗪環", "type": "scaffold_swap"},
        {"name": "add_morpholine", "description": "添加嗎啉環", "type": "scaffold_swap"},
        {"name": "add_pyrrolidine", "description": "添加吡咯烷環", "type": "scaffold_swap"},
        {"name": "add_tetrahydrofuran", "description": "添加四氫呋喃環", "type": "scaffold_swap"},
        {"name": "add_tetrahydropyran", "description": "添加四氫吡喃環", "type": "scaffold_swap"},
    ]
    
    # 藥物常見骨架 (5種)
    drug_scaffolds = [
        {"name": "add_biphenyl", "description": "添加聯苯結構", "type": "scaffold_swap"},
        {"name": "add_pyrimidinone", "description": "添加嘧啶酮結構", "type": "scaffold_swap"},
        {"name": "add_triazine", "description": "添加三嗪環", "type": "scaffold_swap"},
        {"name": "add_triazole", "description": "添加三唑環", "type": "scaffold_swap"},
        {"name": "add_tetrazole", "description": "添加四唑環", "type": "scaffold_swap"},
    ]
    
    # 官能基添加 (3種)
    functional_groups = [
        {"name": "add_hydroxyl", "description": "添加羥基 (-OH)", "type": "functional_group"},
        {"name": "add_amino", "description": "添加氨基 (-NH2)", "type": "functional_group"},
        {"name": "add_carboxyl", "description": "添加羧基 (-COOH)", "type": "functional_group"},
    ]
    
    # 結構調整 (6種)
    structure_modifications = [
        {"name": "reduce_molecular_weight", "description": "減少分子量", "type": "modification"},
        {"name": "heteroatom_exchange", "description": "雜原子交換", "type": "modification"},
        {"name": "ring_formation", "description": "成環反應", "type": "modification"},
        {"name": "ring_opening", "description": "開環反應", "type": "modification"},
        {"name": "chain_extension", "description": "鏈延長", "type": "modification"},
        {"name": "chain_shortening", "description": "鏈縮短", "type": "modification"},
    ]
    
    # 合併所有 actions
    actions.extend(aromatic_single)
    actions.extend(aromatic_fused)
    actions.extend(saturated_rings)
    actions.extend(drug_scaffolds)
    actions.extend(functional_groups)
    actions.extend(structure_modifications)
    
    return actions