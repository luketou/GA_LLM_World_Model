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


def sample(parent_smiles: str, k: int, scaffold_priority_factor: float = 1.5) -> List[Dict[str, Any]]:
    """
    從粗操作集中隨機選 k 種操作，可選擇性提高骨架抽換操作的優先級。

    Args:
        parent_smiles: 父分子 SMILES (目前在此抽樣邏輯中未使用，但保留以維持API一致性).
        k: 要選擇的操作數量.
        scaffold_priority_factor: 骨架抽換操作的優先級因子。
                                     - 大於 1.0 會增加選擇骨架抽換的機率。
                                       例如，2.0 表示骨架抽換的選擇機率約為其他類型操作的兩倍。
                                     - 設為 1.0 則近似於均等機率抽樣 (類似原始的 random.sample)。
                                     - 建議值介於 1.0 到 3.0 之間開始嘗試。
    Returns:
        選擇的粗操作列表.
    """
    if not COARSE or k <= 0:
        return []

    num_to_sample = min(k, len(COARSE))
    if num_to_sample <= 0:
        return []

    # 確保優先級因子有效
    if scaffold_priority_factor <= 0:
        scaffold_priority_factor = 1.0

    # 創建帶權重的操作列表副本，用於抽樣
    # 這樣做是為了在抽樣過程中可以移除已選中的項目，模擬無放回抽樣
    temp_weighted_actions = []
    for i, action in enumerate(COARSE):
        weight = scaffold_priority_factor if action.get("type") == "scaffold_swap" else 1.0
        temp_weighted_actions.append({"action": action, "weight": weight, "id": i}) # id 用於唯一標識

    selected_actions_final = []
    
    # 進行 k 次加權抽樣（模擬無放回）
    for _ in range(num_to_sample):
        if not temp_weighted_actions: # 如果所有操作都已被選中
            break

        # 準備當前輪次的抽樣列表和對應權重
        actions_for_choice = [item["action"] for item in temp_weighted_actions]
        weights_for_choice = [item["weight"] for item in temp_weighted_actions]

        # 檢查權重是否有效 (例如，總和是否為0)
        if not weights_for_choice or sum(weights_for_choice) == 0:
            # 如果所有剩餘權重都為0，則退回到均勻隨機抽樣
            if actions_for_choice:
                chosen_action_object = random.choice(actions_for_choice)
            else:
                break # 沒有更多操作可選
        else:
            # random.choices 返回一個列表 (即使 k=1)，我們取第一個元素
            chosen_action_object = random.choices(actions_for_choice, weights=weights_for_choice, k=1)[0]
        
        selected_actions_final.append(chosen_action_object)

        # 從 temp_weighted_actions 中移除已選中的操作，以模擬無放回抽樣
        index_to_remove = -1
        for i, item in enumerate(temp_weighted_actions):
            if item["action"] == chosen_action_object: 
                index_to_remove = i
                break
        
        if index_to_remove != -1:
            temp_weighted_actions.pop(index_to_remove)
        else:
            if temp_weighted_actions: 
                 temp_weighted_actions.pop(random.randrange(len(temp_weighted_actions)))

    return selected_actions_final