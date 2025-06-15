"""
MCTS Engine - 核心引擎
- propose_actions: 根據 depth 決定 coarse 或 fine 操作
- update_batch: 更新 MCTS 樹、Neo4j KG、最佳節點
- select_child: 依 UCT 選擇最佳子節點
"""
from typing import List, Dict, Optional
from .node import Node
from .uct import uct
from .progressive_widening import allow_expand
from actions.coarse_actions import sample as coarse_sample
from actions.fine_actions import expand as fine_expand
from kg.kg_store import KGStore


class MCTSEngine:
    """MCTS 核心引擎"""
    
    def __init__(self, kg: KGStore, max_depth: int):
        self.kg = kg
        self.root = Node(smiles="ROOT", depth=0)
        self.best = self.root
        self.max_depth = max_depth
        self.epoch = 0
        self.nodes = {"ROOT": self.root}  # 快速查找節點

    # ----- external API -----
    def propose_actions(self, parent_smiles: str, depth: int, k: int) -> List[Dict]:
        """
        根據 depth 決定 coarse 或 fine 操作
        
        Args:
            parent_smiles: 父分子 SMILES
            depth: 當前深度
            k: 要產生的動作數量
            
        Returns:
            動作列表
        """
        if depth < 5:   # coarse 層：使用粗粒度操作
            return coarse_sample(parent_smiles, k)
        else:   # fine 層：使用微調操作
            # Progressive Widening 決定 unlock factor
            parent = self._get_or_create_node(parent_smiles, depth)
            unlock_factor = parent.visits ** 0.6 if parent.visits > 0 else 0.1
            return fine_expand(parent_smiles, unlock_factor, top_k=k)

    def update_batch(self,
                     parent_smiles: str,
                     batch_smiles: List[str],
                     scores: List[float],
                     advantages: List[float]):
        """
        更新 MCTS 樹、Neo4j KG、最佳節點
        
        Args:
            parent_smiles: 父分子 SMILES
            batch_smiles: 生成的分子列表
            scores: 對應的分數列表
            advantages: 對應的優勢值列表
        """
        parent = self._get_or_create_node(parent_smiles, 0)  # depth will be updated
        baseline = sum(scores) / len(scores) if scores else 0.0
        
        # 更新子節點
        for smiles, score, advantage in zip(batch_smiles, scores, advantages):
            child = self._get_or_create_node(smiles, parent.depth + 1)
            
            # 更新節點統計
            child.visits += 1
            child.total_score += score
            child.advantage = advantage
            child.regret = baseline - score
            
            # 添加到父節點的子節點
            parent.children[smiles] = child
            
            # 寫入知識圖譜
            self.kg.create_molecule(
                smiles=smiles,
                score=score,
                advantage=advantage,
                regret=child.regret,
                epoch=self.epoch
            )
            
            # 更新最佳節點
            if child.mean_score > self.best.mean_score:
                self.best = child
        
        # 更新父節點訪問次數
        parent.visits += len(batch_smiles)
        self.epoch += 1

    def select_child(self, parent_smiles: str) -> Optional[Node]:
        """
        依 UCT 選擇最佳子節點
        
        Args:
            parent_smiles: 父分子 SMILES
            
        Returns:
            選中的子節點，如果沒有則返回 None
        """
        parent = self.nodes.get(parent_smiles)
        if not parent or not parent.children:
            return None
        
        best_value = -float('inf')
        best_child = None
        
        for child in parent.children.values():
            if child.visits == 0:
                # 優先選擇未訪問的節點
                return child
            
            uct_value = uct(
                parent_n=parent.visits,
                child_n=child.visits,
                q=child.mean_score,
                adv=child.advantage
            )
            
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        
        return best_child

    def _get_or_create_node(self, smiles: str, depth: int) -> Node:
        """獲取或創建節點"""
        if smiles not in self.nodes:
            self.nodes[smiles] = Node(smiles=smiles, depth=depth)
        return self.nodes[smiles]

    def get_stats(self) -> Dict:
        """獲取引擎統計信息"""
        return {
            "total_nodes": len(self.nodes),
            "epochs": self.epoch,
            "best_score": self.best.mean_score,
            "best_smiles": self.best.smiles,
            "max_depth": self.max_depth
        }