"""
MCTS (Monte Carlo Tree Search) 模組
包含以下組件:
- Node: MCTS 節點實現
- UCTSelector: UCT 選擇策略
- ProgressiveWidening: 漸進拓寬策略
- MCTSEngine: 主要的 MCTS 引擎
- SearchStrategies: 搜索策略集合
- TreeAnalytics: 樹分析器
- TreeManipulator: 樹操作器
- LLMGuidedActionSelector: LLM引導的軌跡感知動作選擇器
"""
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add mcts directory to path
mcts_dir = os.path.dirname(os.path.abspath(__file__))
if mcts_dir not in sys.path:
    sys.path.insert(0, mcts_dir)

try:
    # Use absolute imports
    from mcts.node import Node
    from mcts.uct import UCTSelector
    from mcts.progressive_widening import ProgressiveWidening
    from mcts.search_strategies import SearchStrategies
    from mcts.tree_analytics import TreeAnalytics
    from mcts.tree_manipulator import TreeManipulator
    from mcts.mcts_engine import MCTSEngine
    from mcts.llm_guided_selector import LLMGuidedActionSelector, create_llm_guided_selector
    
    print("MCTS modules imported successfully")
    
except ImportError as e:
    print(f"Warning: Some MCTS modules could not be imported: {e}")
    # Create simple fallback implementations
    class Node:
        def __init__(self, smiles: str, depth: int = 0):
            self.smiles = smiles
            self.depth = depth
            self.visits = 0
            self.total_score = 0.0
            self.children = {}
            self.parent = None
            self.is_terminal = False
        
        @property
        def avg_score(self):
            return self.total_score / max(self.visits, 1)
        
        def update(self, score: float):
            self.visits += 1
            self.total_score += score
        
        def add_child(self, child_smiles: str, child_depth: int = None):
            if child_depth is None:
                child_depth = self.depth + 1
            child_node = Node(smiles=child_smiles, depth=child_depth)
            child_node.parent = self
            self.children[child_smiles] = child_node
            return child_node
        
        def is_fully_expanded(self, max_children: int = 30):
            return len(self.children) >= max_children or self.is_terminal
        
        def has_children(self):
            return len(self.children) > 0
    
    UCTSelector = None
    ProgressiveWidening = None
    SearchStrategies = None
    TreeAnalytics = None
    TreeManipulator = None
    MCTSEngine = None
    LLMGuidedActionSelector = None
    create_llm_guided_selector = None

__all__ = [
    'Node',
    'UCTSelector', 
    'ProgressiveWidening',
    'SearchStrategies',
    'MCTSEngine',
    'TreeAnalytics',
    'TreeManipulator',
    'LLMGuidedActionSelector',
    'create_llm_guided_selector'
]