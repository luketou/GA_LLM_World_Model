"""
Neo4j Knowledge-Graph CRUD layer.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class KGConfig:
    uri: str
    user: str
    password: str
    backend: str = "neo4j"
    database: str = "neo4j"
    enabled: bool = True


class MockKGStore:
    """Mock KG store for when Neo4j is not available"""
    def __init__(self, cfg: KGConfig):
        self.molecules = {}
        self.actions = []
        logger.info("Using mock KG store (Neo4j disabled)")
    
    def close(self):
        pass
    
    def create_molecule(self, smiles: str, score: float, advantage: float = 0.0, regret: float = 0.0, epoch: int = 0):
        self.molecules[smiles] = {
            'score': score, 'advantage': advantage, 'regret': regret, 'epoch': epoch
        }
    
    def create_action(self, parent_smiles: str, child_smiles: str, action_type: str, action_params: str, score_delta: float):
        self.actions.append({
            'parent': parent_smiles, 'child': child_smiles, 'type': action_type, 
            'params': action_params, 'delta': score_delta
        })
    
    def top_k(self, k: int) -> List[Dict[str, Any]]:
        sorted_mols = sorted(self.molecules.items(), key=lambda x: x[1]['score'], reverse=True)
        return [{'smiles': smiles, 'score': data['score'], 'visits': 1} 
                for smiles, data in sorted_mols[:k]]
    
    def mark_cold(self, smiles: str):
        if smiles in self.molecules:
            self.molecules[smiles]['cold'] = True

    def get_branch_history(self, target_smiles: str) -> List[Dict]:
        """
        獲取從根節點到目標分子的完整路徑歷史
        
        Args:
            target_smiles: 目標分子的 SMILES 字符串
            
        Returns:
            歷史路徑列表，每個元素包含 {'smiles': str, 'action': str, 'score': float}
        """
        if target_smiles not in self.molecules:
            logger.warning(f"Target SMILES {target_smiles} not found in KG")
            return []
        
        # 回溯路徑
        path = []
        current_smiles = target_smiles
        visited = set()  # 防止無限循環
        
        while current_smiles and current_smiles in self.molecules and current_smiles not in visited:
            visited.add(current_smiles)
            node_data = self.molecules[current_smiles]
            
            # 添加當前節點信息
            path_entry = {
                'smiles': current_smiles,
                'action': node_data.get('action', 'ROOT'),
                'score': node_data.get('score', 0.0)
            }
            path.append(path_entry)
            
            # 移動到父節點
            parent_smiles = node_data.get('parent')
            if parent_smiles == current_smiles or parent_smiles is None:
                break
            current_smiles = parent_smiles
        
        # 反轉路徑，使其從根節點到目標節點
        path.reverse()
        
        logger.debug(f"Branch history for {target_smiles}: {len(path)} nodes")
        return path


class KGStore:
    def __init__(self, cfg: KGConfig):
        # 檢查是否啟用 Neo4j
        if not cfg.enabled:
            raise ConnectionError("Neo4j is disabled")
        
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(cfg.uri,
                                               auth=(cfg.user, cfg.password),
                                               database=cfg.database)
            self.database = cfg.database
            logger.info("Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        self.driver.close()

    # ---------- write ----------
    def create_molecule(self, smiles: str,
                        score: float,
                        advantage: float = 0.0,
                        regret: float = 0.0,
                        epoch: int = 0):
        """創建或更新分子節點，累積分數和訪問次數"""
        q = ("MERGE (m:Molecule {smiles:$s}) "
             "ON CREATE SET m.score=$sc, m.advantage=$ad, m.regret=$re, m.epoch=$ep, "
             "             m.visits=1, m.total_score=$sc, m.cold=false "
             "ON MATCH  SET m.total_score=m.total_score+$sc, "
             "             m.visits=m.visits+1, "
             "             m.score=m.total_score/m.visits, "
             "             m.advantage=$ad, m.regret=$re, m.epoch=$ep")
        with self.driver.session(database=self.database) as s:
            s.run(q, s=smiles, sc=score, ad=advantage, re=regret, ep=epoch)

    def create_action(self, parent_smiles: str, child_smiles: str,
                      action_type: str, action_params: str, score_delta: float):
        """建立父子關係邊並設定操作類型、參數、分數變化"""
        q = ("MATCH (p:Molecule {smiles:$p}), (c:Molecule {smiles:$c}) "
             "MERGE (p)-[r:ACTION {type:$t, params:$pa}]->(c) "
             "SET r.delta=$d, r.timestamp=datetime()")
        with self.driver.session(database=self.database) as s:
            s.run(q, p=parent_smiles, c=child_smiles, t=action_type, pa=action_params, d=score_delta)

    # ---------- query ----------
    def top_k(self, k: int) -> List[Dict[str, Any]]:
        """回傳分數最高的 k 個分子"""
        q = ("MATCH (m:Molecule) "
             "WHERE m.cold IS NULL OR m.cold = false "
             "RETURN m.smiles AS smiles, m.score AS score, m.visits AS visits "
             "ORDER BY m.score DESC LIMIT $k")
        with self.driver.session(database=self.database) as s:
            return [dict(r) for r in s.run(q, k=k)]

    # ---------- pruning mark ----------
    def mark_cold(self, smiles: str):
        """為低潛力節點加上 cold=true 標籤"""
        with self.driver.session(database=self.database) as s:
            s.run("MATCH (m:Molecule {smiles:$s}) SET m.cold=true", s=smiles)

    def get_branch_history(self, target_smiles: str) -> List[Dict]:
        """
        獲取從根節點到目標分子的完整路徑歷史
        
        Args:
            target_smiles: 目標分子的 SMILES 字符串
            
        Returns:
            歷史路徑列表，每個元素包含 {'smiles': str, 'action': str, 'score': float}
        """
        query = """
        MATCH path = (root:Molecule)-[:GENERATES*]->(target:Molecule {smiles: $target_smiles})
        WHERE NOT EXISTS((root)<-[:GENERATES]-())
        WITH nodes(path) as path_nodes, relationships(path) as path_rels
        UNWIND range(0, size(path_nodes)-1) as i
        RETURN 
            path_nodes[i].smiles as smiles,
            CASE WHEN i = 0 THEN 'ROOT' ELSE path_rels[i-1].action END as action,
            path_nodes[i].score as score
        ORDER BY i
        """
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, target_smiles=target_smiles)
                history = []
                for record in result:
                    history.append({
                        'smiles': record['smiles'],
                        'action': record['action'] or 'ROOT',
                        'score': record['score'] or 0.0
                    })
                return history
        except Exception as e:
            print(f"Error getting branch history: {e}")
            return []


def create_kg_store(cfg: KGConfig):
    """Factory function to create appropriate KG store"""
    if not cfg.enabled:
        return MockKGStore(cfg)
    
    try:
        return KGStore(cfg)
    except Exception as e:
        logger.warning(f"Failed to create Neo4j store, using mock: {e}")
        return MockKGStore(cfg)