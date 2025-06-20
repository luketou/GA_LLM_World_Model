"""
MCTS Engine - 核心引擎
- propose_actions: 根據 depth 決定 coarse 或 fine 操作
- update_batch: 更新 MCTS 樹、Neo4j KG、最佳節點
- select_child: 依 UCT 選擇最佳子節點
"""
import logging
from typing import Dict, List, Any, Optional
from .node import Node
from .uct import UCTSelector
from .progressive_widening import ProgressiveWidening

logger = logging.getLogger(__name__)

class MCTSEngine:
    """蒙地卡羅樹搜索引擎"""
    
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
        
        # 使用字典存儲節點，key 為 SMILES 字符串
        self.nodes: Dict[str, Node] = {}
        self.root_smiles: Optional[str] = None
        
        # 初始化策略組件
        self.uct_selector = UCTSelector(c_uct=c_uct)
        self.progressive_widening = ProgressiveWidening()
        
        logger.info(f"MCTSEngine initialized with max_depth={max_depth}, c_uct={c_uct}")
    
    def initialize_root(self, root_smiles: str):
        """初始化根節點"""
        self.root_smiles = root_smiles
        if root_smiles not in self.nodes:
            self.nodes[root_smiles] = Node(smiles=root_smiles, depth=0)
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
    
    def expand(self, parent_smiles: str, actions: List[Dict], batch_size: int = 30) -> List[str]:
        """
        擴展節點
        
        Args:
            parent_smiles: 父節點 SMILES
            actions: 動作列表
            batch_size: 批次大小
            
        Returns:
            List[str]: 生成的子節點 SMILES 列表
        """
        try:
            logger.info(f"Expanding node: {parent_smiles} with {len(actions)} actions")
            
            # 獲取或創建父節點
            parent_node = self.get_or_create_node(parent_smiles)
            
            # 使用漸進拓寬策略調整動作
            if hasattr(self, 'progressive_widening'):
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
    
    def select_best_child_uct(self, parent_smiles: str) -> Optional[str]:
        """
        使用 UCT 公式選擇最佳子節點
        
        Args:
            parent_smiles: 父節點 SMILES
            
        Returns:
            Optional[str]: 最佳子節點 SMILES
        """
        parent_node = self.nodes.get(parent_smiles)
        if not parent_node:
            return None
        
        best_child_node = self.uct_selector.select_best_child(parent_node)
        return best_child_node.smiles if best_child_node else None
    
    def select(self, root_smiles: str) -> Optional[str]:
        """選擇階段：從根節點選擇到葉節點"""
        root_node = self.nodes.get(root_smiles)
        if not root_node:
            return root_smiles
        
        leaf_node = self.uct_selector.select_path_to_leaf(root_node, self.max_depth)
        return leaf_node.smiles
    
    def backpropagate(self, smiles_list: List[str], scores: List[float]):
        """反向傳播分數"""
        if len(smiles_list) != len(scores):
            logger.error(f"Mismatch: {len(smiles_list)} smiles vs {len(scores)} scores")
            return
        
        updated_count = 0
        for smiles, score in zip(smiles_list, scores):
            node = self.nodes.get(smiles)
            if node:
                node.update(score)
                updated_count += 1
                
                # 向上傳播到父節點
                current = node.parent
                decay_factor = 0.9
                while current:
                    current.update(score * decay_factor)
                    current = current.parent
                    decay_factor *= 0.9
                    
        logger.info(f"Backpropagation complete: updated {updated_count}/{len(smiles_list)} nodes")
    
    def get_best_node(self) -> Optional[Node]:
        """獲取最佳節點"""
        if not self.nodes:
            return None
        
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
        """獲取搜索統計信息"""
        total_nodes = len(self.nodes)
        visited_nodes = sum(1 for node in self.nodes.values() if node.visits > 0)
        total_visits = sum(node.visits for node in self.nodes.values())
        
        if visited_nodes > 0:
            avg_score = sum(node.total_score for node in self.nodes.values()) / total_visits
            max_depth = self._calculate_max_depth()
        else:
            avg_score = 0.0
            max_depth = 0
        
        best_node = self.get_best_node()
        best_score = best_node.avg_score if best_node else 0.0
        
        stats = {
            "total_nodes": total_nodes,
            "visited_nodes": visited_nodes,
            "total_visits": total_visits,
            "average_score": avg_score,
            "max_depth": max_depth,
            "best_score": best_score,
            "best_smiles": best_node.smiles if best_node else None
        }
        
        logger.info(f"MCTS Statistics: {stats}")
        return stats
    
    def _calculate_max_depth(self) -> int:
        """計算搜索樹的最大深度"""
        if not self.root_smiles or self.root_smiles not in self.nodes:
            return 0
        
        def get_depth(node: Node, current_depth: int = 0) -> int:
            if not node.children:
                return current_depth
            return max(get_depth(child, current_depth + 1) 
                      for child in node.children.values())
        
        return get_depth(self.nodes[self.root_smiles])
    
    def reset(self):
        """重置 MCTS 引擎"""
        self.nodes.clear()
        self.root_smiles = None
        logger.info("MCTS engine reset")