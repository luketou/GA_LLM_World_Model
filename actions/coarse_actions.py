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


def sample(parent_smiles: str, k: int) -> List[Dict[str, Any]]:
    """從粗操作集中隨機選 k 種操作"""
    return random.sample(COARSE, k=min(k, len(COARSE)))