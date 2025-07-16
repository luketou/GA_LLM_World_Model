"""
Search Strategies Module
處理各種搜索策略，包括基於分數的選擇和其他非UCT策略
"""
from typing import Optional, TYPE_CHECKING
import logging
import random

if TYPE_CHECKING:
    from .node import Node

logger = logging.getLogger(__name__)

class SearchStrategies:
    """搜索策略集合 - 提供多種節點選擇策略"""
    
    @staticmethod
    def select_best_child_by_score(parent_node) -> Optional["Node"]:
        """
        基於分數選擇最佳子節點
        
        Args:
            parent_node: 父節點
            
        Returns:
            Optional[Node]: 最佳子節點，如果沒有則返回 None
        """
        if not parent_node.children:
            return None
        
        best_child = None
        best_score = -float('inf')
        
        for child in parent_node.children.values():
            if child.visits > 0:  # 只考慮已經訪問過的節點
                if child.avg_score > best_score:
                    best_score = child.avg_score
                    best_child = child
        
        if best_child:
            logger.debug(f"Selected best scoring child: {best_child.smiles} (score: {best_score:.4f})")
            return best_child
        
        # 如果沒有訪問過的子節點，隨機選擇一個
        child_smiles = random.choice(list(parent_node.children.keys()))
        selected_child = parent_node.children[child_smiles]
        logger.debug(f"Randomly selected child: {selected_child.smiles}")
        return selected_child
    
    @staticmethod
    def select_most_visited_child(parent_node) -> Optional["Node"]:
        """
        選擇訪問次數最多的子節點
        
        Args:
            parent_node: 父節點
            
        Returns:
            Optional[Node]: 最多訪問的子節點
        """
        if not parent_node.children:
            return None
        
        return max(parent_node.children.values(), key=lambda child: child.visits)
    
    @staticmethod
    def select_random_child(parent_node) -> Optional["Node"]:
        """
        隨機選擇子節點
        
        Args:
            parent_node: 父節點
            
        Returns:
            Optional[Node]: 隨機選擇的子節點
        """
        if not parent_node.children:
            return None
        
        child_smiles = random.choice(list(parent_node.children.keys()))
        return parent_node.children[child_smiles]
    
    @staticmethod
    def select_least_visited_child(parent_node) -> Optional["Node"]:
        """
        選擇訪問次數最少的子節點（探索導向）
        
        Args:
            parent_node: 父節點
            
        Returns:
            Optional[Node]: 最少訪問的子節點
        """
        if not parent_node.children:
            return None
        
        return min(parent_node.children.values(), key=lambda child: child.visits)
    
    @staticmethod
    def select_balanced_child(parent_node, exploration_weight: float = 0.5) -> Optional["Node"]:
        """
        平衡選擇策略：結合分數和探索
        
        Args:
            parent_node: 父節點
            exploration_weight: 探索權重 (0.0 = 純利用, 1.0 = 純探索)
            
        Returns:
            Optional[Node]: 平衡選擇的子節點
        """
        if not parent_node.children:
            return None
        
        children = list(parent_node.children.values())
        
        # 計算每個子節點的平衡分數
        scores = []
        for child in children:
            if child.visits == 0:
                balanced_score = float('inf')  # 未訪問節點優先
            else:
                exploitation_score = child.avg_score
                exploration_score = 1.0 / (child.visits + 1)  # 反比例於訪問次數
                balanced_score = (1 - exploration_weight) * exploitation_score + exploration_weight * exploration_score
            
            scores.append(balanced_score)
        
        # 選擇分數最高的子節點
        best_idx = scores.index(max(scores))
        selected_child = children[best_idx]
        
        logger.debug(f"Balanced selection: {selected_child.smiles} (score: {scores[best_idx]:.4f})")
        return selected_child