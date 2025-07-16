"""
Tree Analytics Module
處理樹結構的統計分析和性能評估
"""
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class TreeAnalytics:
    """樹分析器 - 提供樹結構統計和分析功能"""
    
    @staticmethod
    def get_tree_statistics(root_node) -> Dict[str, Any]:
        """
        獲取以指定節點為根的子樹統計信息
        
        Args:
            root_node: 根節點
            
        Returns:
            Dict[str, Any]: 樹統計信息
        """
        def _traverse(node) -> Dict:
            stats = {
                "total_nodes": 1,
                "visited_nodes": 1 if node.visits > 0 else 0,
                "total_visits": node.visits,
                "max_depth": node.depth,
                "leaf_nodes": 0 if node.children else 1
            }
            
            for child in node.children.values():
                child_stats = _traverse(child)
                stats["total_nodes"] += child_stats["total_nodes"]
                stats["visited_nodes"] += child_stats["visited_nodes"]
                stats["total_visits"] += child_stats["total_visits"]
                stats["max_depth"] = max(stats["max_depth"], child_stats["max_depth"])
                stats["leaf_nodes"] += child_stats["leaf_nodes"]
            
            return stats
        
        return _traverse(root_node)
    
    @staticmethod
    def get_subtree_best_node(root_node):
        """
        獲取子樹中的最佳節點
        
        Args:
            root_node: 根節點
            
        Returns:
            Node: 最佳節點
        """
        best_node = root_node
        best_score = root_node.avg_score if root_node.visits > 0 else -float('inf')
        
        def _traverse(node):
            nonlocal best_node, best_score
            if node.visits > 0 and node.avg_score > best_score:
                best_score = node.avg_score
                best_node = node
            
            for child in node.children.values():
                _traverse(child)
        
        _traverse(root_node)
        return best_node
    
    @staticmethod
    def calculate_tree_depth(root_node) -> int:
        """
        計算以指定節點為根的樹的最大深度
        
        Args:
            root_node: 根節點
            
        Returns:
            int: 最大深度
        """
        if not root_node.children:
            return root_node.depth
        
        return max(TreeAnalytics.calculate_tree_depth(child) for child in root_node.children.values())
    
    @staticmethod
    def get_path_statistics(root_node) -> Dict[str, Any]:
        """
        獲取路徑統計信息
        
        Args:
            root_node: 根節點
            
        Returns:
            Dict[str, Any]: 路徑統計
        """
        paths = []
        
        def _collect_paths(node, current_path):
            current_path.append(node.smiles)
            
            if not node.children:  # 葉節點
                paths.append({
                    "path": current_path.copy(),
                    "length": len(current_path),
                    "final_score": node.avg_score,
                    "total_visits": sum(n.visits for n in current_path if hasattr(n, 'visits'))
                })
            else:
                for child in node.children.values():
                    _collect_paths(child, current_path)
            
            current_path.pop()
        
        _collect_paths(root_node, [])
        
        if not paths:
            return {"avg_path_length": 0, "max_path_length": 0, "total_paths": 0}
        
        path_lengths = [p["length"] for p in paths]
        
        return {
            "total_paths": len(paths),
            "avg_path_length": sum(path_lengths) / len(path_lengths),
            "max_path_length": max(path_lengths),
            "min_path_length": min(path_lengths),
            "best_path_score": max(p["final_score"] for p in paths)
        }