"""
MCTS Engine - 核心引擎
- propose_actions: 根據 depth 決定 coarse 或 fine 操作
- update_batch: 更新 MCTS 樹、Neo4j KG、最佳節點
- select_child: 依 UCT 選擇最佳子節點
"""
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# 使用條件導入以避免循環導入問題
try:
    from .node import Node
    from .uct import UCTSelector
    from .progressive_widening import ProgressiveWidening
    from .search_strategies import SearchStrategies
    from .tree_analytics import TreeAnalytics
    from .tree_manipulator import TreeManipulator
except ImportError as e:
    logger.warning(f"Import error in mcts_engine: {e}")
    # 如果導入失敗，創建簡單的替代類
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

# 導入 actions 模組
try:
    from actions import fine_actions
    from actions import coarse_actions
    ACTIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Actions modules not available: {e}")
    fine_actions = None
    coarse_actions = None
    ACTIONS_AVAILABLE = False

class MCTSEngine:
    """蒙地卡羅樹搜索引擎 - 使用外部 actions 模組"""
    
    def __init__(self, kg_store, max_depth: int, llm_gen, c_uct: float = 1.414):
        """
        初始化 MCTS 引擎
        
        Args:
            kg_store: 知識圖譜存儲
            max_depth: 最大搜索深度
            llm_gen: LLM 生成器
            c_uct: UCT 探索常數
        """
        self.kg = kg_store
        self.max_depth = max_depth
        self.llm_gen = llm_gen
        self.c_uct = c_uct
        
        # 使用字典存儲節點，key 為 SMILES 字符串
        self.nodes: Dict[str, Node] = {}
        self.root_smiles: Optional[str] = None
        self.root = None
        self.best = None
        self.iteration_count = 0
        
        # 初始化策略組件（如果可用）
        try:
            if UCTSelector:
                self.uct_selector = UCTSelector(c_uct=c_uct)
            else:
                self.uct_selector = None
                
            if ProgressiveWidening:
                self.progressive_widening = ProgressiveWidening()
            else:
                self.progressive_widening = None
        except Exception as e:
            logger.warning(f"Error initializing strategy components: {e}")
            self.uct_selector = None
            self.progressive_widening = None
        
        logger.info(f"MCTSEngine initialized with max_depth={max_depth}, c_uct={c_uct}")
        logger.info(f"Actions modules available: {ACTIONS_AVAILABLE}")
    
    def initialize_root(self, root_smiles: str):
        """初始化根節點"""
        self.root_smiles = root_smiles
        if root_smiles not in self.nodes:
            self.nodes[root_smiles] = Node(smiles=root_smiles, depth=0)
        self.root = self.nodes[root_smiles]
        logger.info(f"Root initialized: {root_smiles}")
    
    def get_or_create_node(self, smiles: str, depth: int = 0) -> Node:
        """獲取或創建節點"""
        if smiles not in self.nodes:
            self.nodes[smiles] = Node(smiles=smiles, depth=depth)
        return self.nodes[smiles]
    
    def _get_or_create_node(self, smiles: str, depth: int = 0) -> Node:
        """獲取或創建節點 - 私有方法（向後兼容）"""
        return self.get_or_create_node(smiles, depth)
    
    def get_node(self, smiles: str) -> Optional[Node]:
        """獲取節點（如果存在）"""
        return self.nodes.get(smiles)
    
    def has_node(self, smiles: str) -> bool:
        """檢查節點是否存在"""
        return smiles in self.nodes
    
    def propose_actions(self, parent_smiles: str, depth: int, k_init: int) -> List[Dict[str, Any]]:
        """
        提議動作列表 - 使用外部 actions 模組
        
        COMPLIANCE NOTE: This method delegates to external actions modules
        which maintain strict compliance with Oracle evaluation constraints.
        
        Args:
            parent_smiles: 父節點 SMILES
            depth: 當前深度
            k_init: 初始動作數量
            
        Returns:
            List[Dict[str, Any]]: 動作列表
        """
        try:
            logger.info(f"Proposing actions for {parent_smiles} at depth {depth}")
            
            # 確保父節點存在
            if parent_smiles not in self.nodes:
                self.get_or_create_node(parent_smiles, depth)
            
            # 使用外部 actions 模組
            if ACTIONS_AVAILABLE and fine_actions:
                # 使用 propose_mixed_actions 函數
                actions = fine_actions.propose_mixed_actions(
                    parent_smiles=parent_smiles,
                    depth=depth,
                    k_init=k_init
                )
                
                logger.info(f"Generated {len(actions)} actions using external module")
                return actions
            else:
                # 後備：使用內建的基本動作
                logger.warning("External actions modules not available, using fallback actions")
                return self._get_fallback_actions(parent_smiles, depth, k_init)
                
        except Exception as e:
            logger.error(f"Error proposing actions for {parent_smiles}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return self._get_fallback_actions(parent_smiles, depth, k_init)
    
    def _get_fallback_actions(self, parent_smiles: str, depth: int, k_init: int) -> List[Dict[str, Any]]:
        """
        後備動作生成
        
        COMPLIANCE NOTE: Fallback actions use only basic chemical operations
        without any RDKit property calculations.
        """
        try:
            # 基於深度調整動作數量
            adjusted_k = max(1, min(k_init, 30 - depth * 3))
            
            basic_actions = [
                {"type": "substitute", "name": "add_methyl", "description": "添加甲基", "params": {"smiles": "C"}},
                {"type": "substitute", "name": "add_hydroxyl", "description": "添加羥基", "params": {"smiles": "O"}},
                {"type": "substitute", "name": "add_amino", "description": "添加氨基", "params": {"smiles": "N"}},
                {"type": "substitute", "name": "add_fluorine", "description": "添加氟", "params": {"smiles": "F"}},
                {"type": "substitute", "name": "add_chlorine", "description": "添加氯", "params": {"smiles": "Cl"}},
                {"type": "substitute", "name": "add_ethyl", "description": "添加乙基", "params": {"smiles": "CC"}},
                {"type": "substitute", "name": "add_phenyl", "description": "添加苯基", "params": {"smiles": "c1ccccc1"}},
                {"type": "substitute", "name": "add_carboxyl", "description": "添加羧基", "params": {"smiles": "C(=O)O"}},
            ]
            
            # 隨機選擇動作
            import random
            if adjusted_k <= len(basic_actions):
                selected_actions = random.sample(basic_actions, adjusted_k)
            else:
                # 如果需要更多動作，重複選擇
                selected_actions = basic_actions * ((adjusted_k // len(basic_actions)) + 1)
                selected_actions = selected_actions[:adjusted_k]
            
            logger.info(f"Generated {len(selected_actions)} fallback actions")
            return selected_actions
            
        except Exception as e:
            logger.error(f"Error in fallback actions: {e}")
            return [{"type": "substitute", "name": "add_methyl", "description": "添加甲基", "params": {"smiles": "C"}}]
    
    def expand(self, parent_smiles: str, actions: List[Dict], batch_size: int = 30) -> List[str]:
        """
        擴展節點
        """
        try:
            logger.info(f"Expanding node: {parent_smiles} with {len(actions)} actions")
            
            # 獲取或創建父節點
            parent_node = self.get_or_create_node(parent_smiles)
            
            # 使用漸進拓寬策略調整動作（如果可用）
            if self.progressive_widening:
                actions = self.progressive_widening.adaptive_expansion(parent_node, actions)
            
            # 檢查是否已經完全擴展
            if parent_node.is_fully_expanded(max_children=batch_size):
                logger.info(f"Node {parent_smiles} is already fully expanded")
                return list(parent_node.children.keys())
            
            # 限制動作數量
            if len(actions) > batch_size:
                logger.warning(f"Too many actions ({len(actions)}), limiting to {batch_size}")
                actions = actions[:batch_size]
            
            if not actions:
                logger.warning("No actions provided for expansion")
                return []
            
            # 使用 LLM 生成新的 SMILES
            new_smiles_list = self.llm_gen.generate_batch(parent_smiles, actions)
            
            # 創建子節點
            valid_children = []
            for smiles in new_smiles_list:
                if smiles and smiles != parent_smiles and smiles not in parent_node.children:
                    child_node = parent_node.add_child(smiles, parent_node.depth + 1)
                    self.nodes[smiles] = child_node
                    valid_children.append(smiles)
                    logger.debug(f"Added child: {smiles}")
            
            logger.info(f"Expansion complete: {len(valid_children)} valid children generated")
            return valid_children
            
        except Exception as e:
            logger.error(f"Error during expansion of {parent_smiles}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def select_child(self, parent_smiles: str) -> Optional[Node]:
        """選擇子節點 - 使用 UCTSelector 或搜索策略"""
        parent_node = self.nodes.get(parent_smiles)
        if not parent_node:
            logger.debug(f"Parent node {parent_smiles} not found")
            return None
        
        if not parent_node.children:
            logger.debug(f"No children available for {parent_smiles}")
            return None
        
        # 使用 UCTSelector 進行選擇（如果可用）
        if self.uct_selector:
            selected_child = self.uct_selector.select_best_child(parent_node)
            if selected_child:
                logger.debug(f"UCT selected child: {selected_child.smiles}")
                return selected_child
        
        # 後備策略：使用 SearchStrategies 模組
        if SearchStrategies:
            selected_child = SearchStrategies.select_best_child_by_score(parent_node)
            if selected_child:
                logger.debug(f"Score-based selected child: {selected_child.smiles}")
                return selected_child
        
        # 最終後備：簡單隨機選擇
        import random
        child_smiles = random.choice(list(parent_node.children.keys()))
        return parent_node.children[child_smiles]
    
    def update_batch(self, parent_smiles: str, batch_smiles: List[str], 
                    scores: List[float], advantages: List[float]):
        """批量更新節點"""
        try:
            logger.info(f"Updating batch for {parent_smiles}: {len(batch_smiles)} molecules")
            
            # 更新所有子節點
            for smiles, score, advantage in zip(batch_smiles, scores, advantages):
                node = self.nodes.get(smiles)
                if node:
                    node.update(score)
                    node.advantage = advantage
                    
                    # 更新最佳節點
                    if not self.best or score > self.best.avg_score:
                        self.best = node
                        logger.info(f"New best node: {smiles} with score {score:.4f}")
            
            # 反向傳播分數
            self.backpropagate(batch_smiles, scores)
            
        except Exception as e:
            logger.error(f"Error updating batch: {e}")
    
    def select_best_child_uct(self, parent_smiles: str) -> Optional[str]:
        """使用 UCT 公式選擇最佳子節點"""
        parent_node = self.nodes.get(parent_smiles)
        if not parent_node or not parent_node.children:
            return None
        
        if self.uct_selector:
            best_child_node = self.uct_selector.select_best_child(parent_node)
            return best_child_node.smiles if best_child_node else None
        else:
            # 簡單的隨機選擇作為回退
            import random
            return random.choice(list(parent_node.children.keys()))
    
    def backpropagate(self, smiles_list: List[str], scores: List[float]):
        """反向傳播分數 - 使用簡化的反向傳播邏輯"""
        if len(smiles_list) != len(scores):
            logger.error(f"Mismatch: {len(smiles_list)} smiles vs {len(scores)} scores")
            return
        
        updated_count = 0
        for smiles, score in zip(smiles_list, scores):
            node = self.nodes.get(smiles)
            if node:
                node.update(score)
                
                # 簡化的反向傳播：向上傳播到父節點
                current = node.parent
                decay_factor = 0.9
                while current:
                    current.update(score * decay_factor)
                    current = current.parent
                    decay_factor *= 0.9
                
                updated_count += 1
                    
        logger.info(f"Backpropagation complete: updated {updated_count}/{len(smiles_list)} nodes")
    
    def get_best_node(self) -> Optional[Node]:
        """獲取最佳節點 - 使用 TreeAnalytics 或後備方案"""
        if not self.nodes:
            return None
        
        # 使用 TreeAnalytics 模組（如果可用）
        if self.root and TreeAnalytics:
            return TreeAnalytics.get_subtree_best_node(self.root)
        
        # 後備方案：在所有節點中搜索
        visited_nodes = [node for node in self.nodes.values() if node.visits > 0]
        if not visited_nodes:
            return None
        
        best_node = max(visited_nodes, key=lambda node: node.avg_score)
        logger.info(f"Best node: {best_node.smiles} (score: {best_node.avg_score:.4f}, visits: {best_node.visits})")
        return best_node
    
    def get_best_smiles(self) -> Optional[str]:
        """獲取最佳分子的 SMILES"""
        best_node = self.get_best_node()
        return best_node.smiles if best_node else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """獲取搜索統計信息 - 使用 TreeAnalytics 模組或後備方案"""
        if not self.nodes:
            return {
                "total_nodes": 0,
                "visited_nodes": 0,
                "total_visits": 0,
                "average_score": 0.0,
                "max_depth": 0,
                "best_score": 0.0,
                "best_smiles": None
            }
        
        # 如果有根節點且 TreeAnalytics 可用，使用其樹統計功能
        if self.root and TreeAnalytics:
            tree_stats = TreeAnalytics.get_tree_statistics(self.root)
            best_node = TreeAnalytics.get_subtree_best_node(self.root)
        else:
            # 後備方案：手動計算
            total_nodes = len(self.nodes)
            visited_nodes = sum(1 for node in self.nodes.values() if node.visits > 0)
            total_visits = sum(node.visits for node in self.nodes.values())
            max_depth = max((node.depth for node in self.nodes.values()), default=0)
            
            tree_stats = {
                "total_nodes": total_nodes,
                "visited_nodes": visited_nodes,
                "total_visits": total_visits,
                "max_depth": max_depth,
                "leaf_nodes": sum(1 for node in self.nodes.values() if not node.children)
            }
            
            best_node = self.get_best_node()
        
        # 計算平均分數
        avg_score = 0.0
        if tree_stats["total_visits"] > 0:
            total_score = sum(node.total_score for node in self.nodes.values())
            avg_score = total_score / tree_stats["total_visits"]
        
        stats = {
            "total_nodes": tree_stats["total_nodes"],
            "visited_nodes": tree_stats["visited_nodes"],
            "total_visits": tree_stats["total_visits"],
            "average_score": avg_score,
            "max_depth": tree_stats["max_depth"],
            "leaf_nodes": tree_stats.get("leaf_nodes", 0),
            "best_score": best_node.avg_score if best_node and best_node.visits > 0 else 0.0,
            "best_smiles": best_node.smiles if best_node and best_node.visits > 0 else None
        }
        
        return stats
    
    def _calculate_max_depth(self) -> int:
        """計算搜索樹的最大深度 - 使用 TreeAnalytics 或後備方案"""
        if not self.root:
            return 0
            
        if TreeAnalytics:
            return TreeAnalytics.calculate_tree_depth(self.root)
        else:
            # 後備方案：簡單深度計算
            return max((node.depth for node in self.nodes.values()), default=0)
    
    def prune_tree(self, keep_top_k: int = 10):
        """修剪搜索樹，使用 TreeManipulator 或後備方案"""
        if not self.root:
            return
        
        if TreeManipulator:
            TreeManipulator.prune_tree_recursive(self.root, keep_top_k)
        else:
            # 後備方案：簡單修剪
            def _prune_recursive(node: Node):
                for child in list(node.children.values()):
                    _prune_recursive(child)
                
                # 簡單修剪邏輯
                if len(node.children) > keep_top_k:
                    sorted_children = sorted(
                        node.children.items(),
                        key=lambda x: x[1].avg_score if x[1].visits > 0 else -float('inf'),
                        reverse=True
                    )
                    node.children = dict(sorted_children[:keep_top_k])
            
            _prune_recursive(self.root)
            logger.info(f"Tree pruning complete, kept top {keep_top_k} children per node")
    
    def reset(self):
        """重置 MCTS 引擎"""
        self.nodes.clear()
        self.root_smiles = None
        self.root = None
        self.best = None
        self.iteration_count = 0
        logger.info("MCTS engine reset")