import math
import logging
from typing import List, Dict, Any
from .node import Node

logger = logging.getLogger(__name__)

class ProgressiveWidening:
    """漸進拓寬策略"""
    
    def __init__(self, alpha: float = 0.5, beta: float = 2.0):
        """
        初始化漸進拓寬
        
        Args:
            alpha: 拓寬參數 alpha
            beta: 拓寬參數 beta
        """
        self.alpha = alpha
        self.beta = beta
    
    def calculate_max_children(self, node: Node) -> int:
        """
        根據漸進拓寬公式計算最大子節點數
        
        Args:
            node: 節點
            
        Returns:
            int: 最大子節點數
        """
        if node.visits == 0:
            return 1
        
        # 漸進拓寬公式: max_children = ceil(alpha * visits^beta)
        max_children = math.ceil(self.alpha * (node.visits ** self.beta))
        
        # 設置合理的上下限
        max_children = max(1, min(max_children, 50))
        
        logger.debug(f"Node {node.smiles} (visits={node.visits}): max_children={max_children}")
        return max_children
    
    def should_expand(self, node: Node) -> bool:
        """
        判斷是否應該擴展節點
        
        Args:
            node: 節點
            
        Returns:
            bool: 是否應該擴展
        """
        current_children = len(node.children)
        max_children = self.calculate_max_children(node)
        
        should_expand = current_children < max_children and not node.is_terminal
        
        logger.debug(f"Node {node.smiles}: current_children={current_children}, "
                    f"max_children={max_children}, should_expand={should_expand}")
        
        return should_expand
    
    def get_expansion_budget(self, node: Node) -> int:
        """
        獲取擴展預算（可以添加多少個新子節點）
        
        Args:
            node: 節點
            
        Returns:
            int: 擴展預算
        """
        current_children = len(node.children)
        max_children = self.calculate_max_children(node)
        
        budget = max(0, max_children - current_children)
        
        logger.debug(f"Expansion budget for {node.smiles}: {budget}")
        return budget
    
    def prioritize_actions(self, actions: List[Dict[str, Any]], node: Node) -> List[Dict[str, Any]]:
        """
        根據漸進拓寬策略對動作進行優先級排序
        
        Args:
            actions: 動作列表
            node: 節點
            
        Returns:
            List[Dict[str, Any]]: 排序後的動作列表
        """
        expansion_budget = self.get_expansion_budget(node)
        
        if expansion_budget == 0:
            logger.debug("No expansion budget available")
            return []
        
        # 如果動作數量超過預算，進行篩選
        if len(actions) > expansion_budget:
            # 簡單的優先級策略：隨機選擇或基於複雜度
            import random
            prioritized_actions = random.sample(actions, expansion_budget)
            logger.debug(f"Prioritized {len(prioritized_actions)} actions from {len(actions)}")
            return prioritized_actions
        
        return actions
    
    def adaptive_expansion(self, node: Node, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        自適應擴展：根據節點性能調整擴展策略
        
        Args:
            node: 節點
            actions: 動作列表
            
        Returns:
            List[Dict[str, Any]]: 調整後的動作列表
        """
        # 如果節點表現好，增加探索預算
        if node.avg_score > 0.7 and node.visits > 5:
            bonus_factor = 1.5
            logger.debug(f"High-performing node {node.smiles}, applying bonus factor {bonus_factor}")
        # 如果節點表現差，減少探索預算
        elif node.avg_score < 0.3 and node.visits > 3:
            bonus_factor = 0.5
            logger.debug(f"Low-performing node {node.smiles}, applying penalty factor {bonus_factor}")
        else:
            bonus_factor = 1.0
        
        # 調整動作數量
        adjusted_budget = int(self.get_expansion_budget(node) * bonus_factor)
        adjusted_budget = max(1, min(adjusted_budget, len(actions)))
        
        if adjusted_budget < len(actions):
            # 選擇最有前景的動作
            prioritized_actions = self.prioritize_actions(actions, node)[:adjusted_budget]
            return prioritized_actions
        
        return actions