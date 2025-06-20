"""
Tree Manipulation Module
處理樹結構的修改操作，包括修剪、重組等
"""
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class TreeManipulator:
    """樹操作器 - 提供樹結構修改功能"""
    
    @staticmethod
    def prune_children(node, keep_top_k: int = 10) -> int:
        """
        修剪子節點，只保留分數最高的 k 個
        
        Args:
            node: 要修剪的節點
            keep_top_k: 保留的子節點數量
            
        Returns:
            int: 被修剪的子節點數量
        """
        if len(node.children) <= keep_top_k:
            return 0
        
        # 按分數排序子節點
        sorted_children = sorted(
            node.children.items(),
            key=lambda x: x[1].avg_score if x[1].visits > 0 else -float('inf'),
            reverse=True
        )
        
        # 保留前 k 個
        new_children = dict(sorted_children[:keep_top_k])
        pruned_count = len(node.children) - len(new_children)
        
        node.children = new_children
        logger.info(f"Pruned {pruned_count} children from node {node.smiles}")
        return pruned_count
    
    @staticmethod
    def prune_tree_recursive(root_node, keep_top_k: int = 10) -> int:
        """
        遞歸修剪整個樹
        
        Args:
            root_node: 根節點
            keep_top_k: 每個節點保留的子節點數量
            
        Returns:
            int: 總共修剪的節點數量
        """
        total_pruned = 0
        
        def _prune_recursive(node):
            nonlocal total_pruned
            
            # 先遞歸修剪子節點
            for child in list(node.children.values()):
                _prune_recursive(child)
            
            # 修剪當前節點的子節點
            pruned_count = TreeManipulator.prune_children(node, keep_top_k)
            total_pruned += pruned_count
        
        _prune_recursive(root_node)
        logger.info(f"Tree pruning complete: {total_pruned} total nodes pruned")
        return total_pruned
    
    @staticmethod
    def remove_low_performing_subtrees(root_node, score_threshold: float = 0.1) -> int:
        """
        移除低性能子樹
        
        Args:
            root_node: 根節點
            score_threshold: 分數閾值，低於此值的子樹將被移除
            
        Returns:
            int: 移除的子樹數量
        """
        removed_count = 0
        
        def _remove_recursive(node):
            nonlocal removed_count
            
            # 收集要移除的子節點
            children_to_remove = []
            
            for child_smiles, child in node.children.items():
                if child.visits > 0 and child.avg_score < score_threshold:
                    children_to_remove.append(child_smiles)
                else:
                    # 遞歸檢查子節點
                    _remove_recursive(child)
            
            # 移除低性能子節點
            for child_smiles in children_to_remove:
                del node.children[child_smiles]
                removed_count += 1
                logger.debug(f"Removed low-performing subtree: {child_smiles}")
        
        _remove_recursive(root_node)
        logger.info(f"Removed {removed_count} low-performing subtrees")
        return removed_count
    
    @staticmethod
    def balance_tree(root_node, max_depth_variance: int = 2) -> int:
        """
        平衡樹結構，減少深度差異
        
        Args:
            root_node: 根節點
            max_depth_variance: 允許的最大深度差異
            
        Returns:
            int: 平衡操作數量
        """
        operations = 0
        
        def _calculate_subtree_depths(node) -> Tuple[int, int]:
            """計算子樹的最小和最大深度"""
            if not node.children:
                return node.depth, node.depth
            
            min_depth = float('inf')
            max_depth = 0
            
            for child in node.children.values():
                child_min, child_max = _calculate_subtree_depths(child)
                min_depth = min(min_depth, child_min)
                max_depth = max(max_depth, child_max)
            
            return min_depth, max_depth
        
        def _balance_recursive(node):
            nonlocal operations
            
            if not node.children:
                return
            
            min_depth, max_depth = _calculate_subtree_depths(node)
            depth_variance = max_depth - min_depth
            
            if depth_variance > max_depth_variance:
                # 找到過深的分支並修剪
                for child in list(node.children.values()):
                    child_min, child_max = _calculate_subtree_depths(child)
                    if child_max - min_depth > max_depth_variance:
                        # 對過深的子樹進行修剪
                        TreeManipulator.prune_children(child, keep_top_k=5)
                        operations += 1
            
            # 遞歸處理子節點
            for child in node.children.values():
                _balance_recursive(child)
        
        _balance_recursive(root_node)
        logger.info(f"Tree balancing complete: {operations} operations performed")
        return operations