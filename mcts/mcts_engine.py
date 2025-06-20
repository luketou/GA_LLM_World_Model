"""
MCTS Engine - 核心引擎
- propose_actions: 根據 depth 決定 coarse 或 fine 操作
- update_batch: 更新 MCTS 樹、Neo4j KG、最佳節點
- select_child: 依 UCT 選擇最佳子節點
"""
import random
import math
import logging
from typing import List, Dict, Optional
from .node import Node
from .uct import uct
from .progressive_widening import allow_expand
from actions.coarse_actions import sample as coarse_sample, get_all_actions as get_coarse_actions
from actions.fine_actions import expand as fine_expand, get_all_actions as get_fine_actions
from kg.kg_store import KGStore
from llm.generator import LLMGenerator

logger = logging.getLogger(__name__)

class MCTSEngine:
    """MCTS 核心引擎"""
    
    def __init__(self, kg_store: KGStore, max_depth: int, llm_gen: LLMGenerator):
        self.kg = kg_store
        self.root = Node(smiles="ROOT", depth=0)
        self.best = self.root
        self.max_depth = max_depth
        self.epoch = 0
        self.nodes = {"ROOT": self.root}  # 快速查找節點
        self.llm_gen = llm_gen
        self.available_actions = get_coarse_actions() + get_fine_actions()

    def add_node(self, smiles: str, parent: str = None, action: str = None):
        if smiles not in self.nodes:
            self.nodes[smiles] = {'parent': parent, 'action': action, 'visits': 0, 'value': 0.0, 'children': []}

    # ----- external API -----
    def propose_actions(self, parent_smiles: str, depth: int, k: int) -> List[Dict]:
        """
        使用 LLM 根據當前分支的歷史紀錄來選擇最佳動作。
        """
        if parent_smiles == "ROOT":
            # 對於根節點，由於沒有歷史紀錄，使用原始的隨機抽樣方法
            logger.info("Root node: using random coarse action sampling.")
            return coarse_sample(k=k)

        logger.info(f"Proposing actions for {parent_smiles} using LLM guidance.")
        
        # 1. 從 KG 獲取分支歷史
        try:
            history = self.kg.get_branch_history(parent_smiles)
        except Exception as e:
            logger.error(f"Error getting branch history: {e}")
            history = []
        
        if not history or len(history) < 2:  # 需要至少有根節點和當前節點
            logger.warning(f"Insufficient history for {parent_smiles}. Falling back to rule-based sampling.")
            # 如果歷史不足，退回至原始的規則式邏輯
            if depth < 5:
                return coarse_sample(k=k)
            else:
                return fine_expand(parent_smiles, k=k)

        # 2. 呼叫 LLM 選擇 actions
        try:
            selected_action_names = self.llm_gen.select_actions(
                parent_smiles=parent_smiles,
                history=history,
                available_actions=self.available_actions,
                k=k
            )
            
            if not selected_action_names:
                raise ValueError("LLM returned empty action list")
            
            # 3. 根據 LLM 回傳的名稱，篩選出對應的 action 物件
            action_map = {action['name']: action for action in self.available_actions}
            selected_actions = [action_map[name] for name in selected_action_names if name in action_map]
            
            if not selected_actions:
                raise ValueError("No valid actions found from LLM selection")
            
            logger.info(f"LLM selected {len(selected_actions)} actions: {[a['name'] for a in selected_actions]}")
            return selected_actions

        except Exception as e:
            logger.error(f"LLM action selection failed: {e}. Falling back to rule-based sampling.")
            # 如果 LLM 選擇失敗，退回至原始的規則式邏輯
            if depth < 5:
                return coarse_sample(k=k)
            else:
                return fine_expand(parent_smiles, k=k)

    def update_batch(self, parent_smiles: str, batch_smiles: List[str], scores: List[float], advantages: List[float]):
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