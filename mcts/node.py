from dataclasses import dataclass, field
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class Node:
    smiles: str
    depth: int = 0
    visits: int = 0
    total_score: float = 0.0
    advantage: float = 0.0
    regret: float = 0.0
    children: Dict[str, "Node"] = field(default_factory=dict)
    parent: Optional["Node"] = None
    is_terminal: bool = False

    @property
    def mean_score(self) -> float:
        """平均分數"""
        return self.total_score / self.visits if self.visits else 0.0
    
    @property
    def avg_score(self) -> float:
        """平均分數的別名，保持向後兼容"""
        return self.mean_score
    
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
    
    def update(self, score: float):
        """更新節點統計"""
        self.visits += 1
        self.total_score += score
        logger.debug(f"Updated node {self.smiles}: visits={self.visits}, avg_score={self.avg_score:.4f}")
    
    def is_fully_expanded(self, max_children: int = 30) -> bool:
        """檢查是否已完全擴展"""
        return len(self.children) >= max_children or self.is_terminal
    
    def has_children(self) -> bool:
        """檢查是否有子節點"""
        return len(self.children) > 0
    
    def get_child(self, child_smiles: str) -> Optional["Node"]:
        """獲取指定的子節點"""
        return self.children.get(child_smiles)
    
    def remove_child(self, child_smiles: str) -> bool:
        """移除子節點"""
        if child_smiles in self.children:
            del self.children[child_smiles]
            return True
        return False
    
    def get_best_child(self) -> Optional["Node"]:
        """獲取最佳子節點（基於平均分數）"""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: child.avg_score)
    
    def get_path_to_root(self) -> list:
        """獲取到根節點的路徑"""
        path = []
        current = self
        while current:
            path.append(current.smiles)
            current = current.parent
        return path[::-1]  # 反轉以從根開始
    
    def calculate_depth(self) -> int:
        """計算實際深度"""
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth
    
    def __str__(self) -> str:
        return f"Node(smiles={self.smiles}, depth={self.depth}, visits={self.visits}, score={self.avg_score:.4f})"
    
    def __repr__(self) -> str:
        return self.__str__()