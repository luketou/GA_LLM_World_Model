"""
MCTS Engine - 核心引擎
- propose_actions: 根據 depth 決定 coarse 或 fine 操作
- update_batch: 更新 MCTS 樹、Neo4j KG、最佳節點
- select_child: 依 UCT 選擇最佳子節點
"""
import random
import math
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
        baseline = sum(scores) / len(scores) if scores else 0.0 # Average score for all molecules
        
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
        使用改進的 UCT 策略選擇最佳子節點
        
        Args:
            parent_smiles: 父分子 SMILES
            
        Returns:
            選中的子節點，如果沒有則返回 None
        """
        parent = self.nodes.get(parent_smiles)
        if not parent or not parent.children:
            return None
        
        # 1. 收集未訪問的節點
        unvisited = [child for child in parent.children.values() if child.visits == 0]
        
        # 2. 優先探索未訪問節點（隨機選擇以增加多樣性）
        if unvisited:
            return random.choice(unvisited)
        
        # 3. 檢查是否需要回溯（所有子節點表現都很差）
        if self._should_backtrack(parent):
            return None  # 觸發回溯策略
        
        # 4. 使用改進的 UCT 選擇已訪問節點
        best_value = -float('inf')
        best_children = []  # 處理相同分數的情況
        
        for child in parent.children.values():
            uct_value = self._calculate_robust_uct(parent, child)
            
            if uct_value > best_value:
                best_value = uct_value
                best_children = [child]
            elif abs(uct_value - best_value) < 1e-6:  # 處理浮點數精度
                best_children.append(child)
        
        # 5. 如果有多個最佳候選，隨機選擇
        return random.choice(best_children) if best_children else None

    def _calculate_robust_uct(self, parent: Node, child: Node) -> float:
        """
        計算穩健的 UCT 值
        """
        # 添加小的常數避免除零
        epsilon = 1e-9
        
        # 使用改進的 UCT 公式
        q_value = child.mean_score
        
        # 探索項：使用對數以避免數值不穩定
        exploration = math.sqrt(2 * math.log(parent.visits + epsilon) / (child.visits + epsilon))
        
        # 優勢項：考慮最近的表現
        advantage_weight = 0.3
        advantage_term = advantage_weight * child.advantage
        
        # 多樣性獎勵：鼓勵探索不同類型的分子
        diversity_bonus = self._calculate_diversity_bonus(child)
        
        # 深度懲罰：避免過深探索
        depth_penalty = 0.01 * child.depth
        
        return q_value + exploration + advantage_term + diversity_bonus - depth_penalty

    def _calculate_diversity_bonus(self, child: Node) -> float:
        """
        計算多樣性獎勵，鼓勵探索化學空間的不同區域
        """
        # 這裡可以基於分子指紋、結構相似性等計算
        # 簡化版本：基於訪問頻率給予獎勵
        visit_ratio = child.visits / (self.epoch + 1)
        return 0.1 * (1 - visit_ratio)  # 訪問較少的節點獲得獎勵

    def _should_backtrack(self, parent: Node) -> bool:
        """
        判斷是否應該回溯到祖先節點
        
        回溯條件：
        1. 連續 N 次子節點的平均優勢 < 閾值
        2. 所有子節點的分數都低於歷史最佳
        3. 探索深度過深且沒有改善
        """
        if not parent.children:
            return False
        
        # 條件1：檢查最近的表現
        recent_advantages = [child.advantage for child in parent.children.values() 
                            if child.visits > 0]
        if recent_advantages:
            avg_advantage = sum(recent_advantages) / len(recent_advantages)
            if avg_advantage < -0.5 and len(recent_advantages) >= 3:
                return True
        
        # 條件2：所有子節點都明顯低於最佳分數
        if self.best.mean_score > 0:
            max_child_score = max(child.mean_score for child in parent.children.values())
            if max_child_score < 0.7 * self.best.mean_score:
                return True
        
        # 條件3：深度過深且無改善
        if parent.depth > 10:
            recent_improvements = [child.mean_score - parent.mean_score 
                                 for child in parent.children.values() if child.visits > 0]
            if all(improvement <= 0 for improvement in recent_improvements):
                return True
        
        return False

    def select_child_with_backtrack(self, parent_smiles: str) -> Optional[Node]:
        """
        帶回溯功能的子節點選擇
        """
        # 嘗試正常選擇
        selected_child = self.select_child(parent_smiles)
        
        if selected_child is None:
            # 觸發回溯策略
            return self._backtrack_to_ancestor(parent_smiles)
        
        return selected_child

    def _backtrack_to_ancestor(self, current_smiles: str) -> Optional[Node]:
        """
        回溯到表現較好的祖先節點
        """
        current = self.nodes.get(current_smiles)
        if not current:
            return None
        
        # 尋找最近的有潛力的祖先
        best_ancestor = None
        best_potential = -float('inf')
        
        # 遍歷祖先節點（這裡需要維護父節點引用）
        # 簡化版本：回溯到根節點附近的高分節點
        for smiles, node in self.nodes.items():
            if (node.depth < current.depth - 2 and 
                node.mean_score > current.mean_score and
                len(node.children) < 8):  # 還有探索空間
                
                potential = node.mean_score + 0.1 * (current.depth - node.depth)
                if potential > best_potential:
                    best_potential = potential
                    best_ancestor = node
        
        return best_ancestor

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