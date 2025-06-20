from dataclasses import dataclass, field
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class Node:
    """
    MCTS 節點類 - 僅包含節點數據結構和基本操作
    
    職責範圍：
    - 節點數據管理（屬性存取、更新）
    - 基本的父子關係操作
    - 節點狀態查詢
    """
    smiles: str
    depth: int = 0
    visits: int = 0
    total_score: float = 0.0
    advantage: float = 0.0
    regret: float = 0.0
    children: Dict[str, "Node"] = field(default_factory=dict)
    parent: Optional["Node"] = None
    is_terminal: bool = False

    # ========== 核心屬性計算 ==========
    
    @property
    def mean_score(self) -> float:
        """平均分數"""
        return self.total_score / self.visits if self.visits else 0.0
    
    @property
    def avg_score(self) -> float:
        """平均分數的別名，保持向後兼容"""
        return self.mean_score
    
    # ========== 基本節點操作 ==========
    
    def update(self, score: float):
        """更新節點統計"""
        self.visits += 1
        self.total_score += score
        logger.debug(f"Updated node {self.smiles}: visits={self.visits}, avg_score={self.avg_score:.4f}")
    
    def add_child(self, child_smiles: str, child_depth: int = None) -> "Node":
        """添加子節點"""
        if child_depth is None:
            child_depth = self.depth + 1
            
        child_node = Node(
            smiles=child_smiles, 
            depth=child_depth,
            parent=self
        )
        self.children[child_smiles] = child_node
        return child_node
    
    def remove_child(self, child_smiles: str) -> bool:
        """移除子節點"""
        if child_smiles in self.children:
            del self.children[child_smiles]
            return True
        return False
    
    def get_child(self, child_smiles: str) -> Optional["Node"]:
        """獲取指定的子節點"""
        return self.children.get(child_smiles)
    
    # ========== 節點狀態查詢 ==========
    
    def has_children(self) -> bool:
        """檢查是否有子節點"""
        return len(self.children) > 0
    
    def is_fully_expanded(self, max_children: int = 30) -> bool:
        """檢查是否已完全擴展"""
        return len(self.children) >= max_children or self.is_terminal
    
    def is_leaf(self) -> bool:
        """檢查是否為葉節點"""
        return not self.children
    
    def is_root(self) -> bool:
        """檢查是否為根節點"""
        return self.parent is None
    
    # ========== 路徑和深度操作 ==========
    
    def get_path_to_root(self) -> List[str]:
        """獲取到根節點的路徑"""
        path = []
        current = self
        while current:
            path.append(current.smiles)
            current = current.parent
        return path[::-1]  # 反轉以從根開始
    
    def calculate_actual_depth(self) -> int:
        """計算實際深度（從根節點計算）"""
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth
    
    # ========== 基本分析方法 ==========
    
    def get_best_child(self) -> Optional["Node"]:
        """獲取最佳子節點（基於平均分數） - 簡單實現"""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.avg_score)
    
    def get_child_count(self) -> int:
        """獲取子節點數量"""
        return len(self.children)
    
    def get_sibling_count(self) -> int:
        """獲取兄弟節點數量"""
        if not self.parent:
            return 0
        return len(self.parent.children) - 1  # 排除自己
    
    # ========== 序列化和表示 ==========
    
    def to_dict(self, include_children: bool = False) -> Dict:
        """轉換為字典表示"""
        result = {
            "smiles": self.smiles,
            "depth": self.depth,
            "visits": self.visits,
            "total_score": self.total_score,
            "avg_score": self.avg_score,
            "advantage": self.advantage,
            "regret": self.regret,
            "is_terminal": self.is_terminal,
            "child_count": len(self.children)
        }
        
        if include_children:
            result["children"] = {
                smiles: child.to_dict(include_children=False) 
                for smiles, child in self.children.items()
            }
        
        return result
    
    def __str__(self) -> str:
        return f"Node(smiles={self.smiles}, depth={self.depth}, visits={self.visits}, score={self.avg_score:.4f})"
    
    def __repr__(self) -> str:
        return self.__str__()