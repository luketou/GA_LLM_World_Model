"""
Neo4j Knowledge-Graph CRUD layer.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
from neo4j import GraphDatabase


@dataclass
class KGConfig:
    uri: str
    user: str
    password: str
    backend: str = "neo4j"
    database: str = "neo4j"


class KGStore:
    def __init__(self, cfg: KGConfig):
        self.driver = GraphDatabase.driver(cfg.uri,
                                           auth=(cfg.user, cfg.password),
                                           database=cfg.database)
        self.database = cfg.database

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