from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
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
    - 動作歷史追蹤
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
    
    # 新增：動作歷史追蹤
    generating_action: Optional[Dict[str, Any]] = None  # 生成此節點的動作
    action_effects: Dict[str, Any] = field(default_factory=dict)  # 動作效果記錄
    
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
        
        # 更新動作效果記錄
        if self.generating_action:
            self.action_effects['score_improvement'] = score - (self.parent.avg_score if self.parent and self.parent.visits > 0 else 0.0)
            self.action_effects['current_score'] = score
            
        logger.debug(f"Updated node {self.smiles}: visits={self.visits}, avg_score={self.avg_score:.4f}")
    
    def add_child(self, child_smiles: str, child_depth: int = None, generating_action: Dict[str, Any] = None) -> "Node":
        """添加子節點"""
        if child_depth is None:
            child_depth = self.depth + 1
            
        child_node = Node(
            smiles=child_smiles, 
            depth=child_depth,
            parent=self,
            generating_action=generating_action
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
    
    # ========== 動作歷史追蹤方法 ==========
    
    def get_action_history(self) -> List[Dict[str, Any]]:
        """獲取從根節點到當前節點的動作歷史"""
        history = []
        current = self
        
        while current and current.generating_action:
            action_record = {
                'action': current.generating_action,
                'depth': current.depth,
                'smiles': current.smiles,
                'parent_smiles': current.parent.smiles if current.parent else None,
                'score_improvement': current.action_effects.get('score_improvement', 0.0),
                'current_score': current.action_effects.get('current_score', 0.0),
                'visits': current.visits
            }
            history.append(action_record)
            current = current.parent
            
        return history[::-1]  # 反轉以從根開始
    
    def get_recent_actions(self, n: int = 3) -> List[Dict[str, Any]]:
        """獲取最近的 n 個動作"""
        history = self.get_action_history()
        return history[-n:] if history else []
    
    def get_action_trajectory_summary(self) -> Dict[str, Any]:
        """獲取動作軌跡摘要，用於LLM上下文"""
        history = self.get_action_history()
        
        if not history:
            return {
                'total_actions': 0,
                'action_types': [],
                'score_trend': 'unknown',
                'recent_actions': []
            }
        
        # 統計動作類型
        action_types = [action['action'].get('type', 'unknown') for action in history]
        action_type_counts = {}
        for action_type in action_types:
            action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
        
        # 分析分數趨勢
        scores = [action.get('current_score', 0.0) for action in history if action.get('current_score') is not None]
        score_trend = 'unknown'
        if len(scores) >= 2:
            recent_trend = scores[-3:] if len(scores) >= 3 else scores
            if len(recent_trend) >= 2:
                if recent_trend[-1] > recent_trend[0]:
                    score_trend = 'improving'
                elif recent_trend[-1] < recent_trend[0]:
                    score_trend = 'declining'
                else:
                    score_trend = 'stable'
        
        return {
            'total_actions': len(history),
            'action_types': list(action_type_counts.keys()),
            'action_type_counts': action_type_counts,
            'score_trend': score_trend,
            'recent_actions': self.get_recent_actions(3),
            'current_depth': self.depth,
            'total_visits': self.visits,
            'avg_score': self.avg_score
        }
    
    def get_successful_action_patterns(self) -> List[Dict[str, Any]]:
        """獲取成功的動作模式（分數改善的動作）"""
        history = self.get_action_history()
        successful_actions = []
        
        for action_record in history:
            score_improvement = action_record.get('score_improvement', 0.0)
            if score_improvement > 0.01:  # 閾值：顯著改善
                successful_actions.append({
                    'action': action_record['action'],
                    'improvement': score_improvement,
                    'context': {
                        'depth': action_record['depth'],
                        'parent_smiles': action_record['parent_smiles']
                    }
                })
        
        # 按改善程度排序
        successful_actions.sort(key=lambda x: x['improvement'], reverse=True)
        return successful_actions
    
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