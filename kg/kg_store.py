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


class KGStore:
    def __init__(self, cfg: KGConfig):
        self.driver = GraphDatabase.driver(cfg.uri,
                                           auth=(cfg.user, cfg.password))

    def close(self):
        self.driver.close()

    # ---------- write ----------
    def create_molecule(self, smiles: str,
                        score: float,
                        adv: float,
                        regret: float,
                        epoch: int):
        q = ("MERGE (m:Molecule {smiles:$s}) "
             "ON CREATE SET m.score=$sc,m.adv=$ad,m.regret=$re,m.epoch=$ep,"
             "             m.visits=1 "
             "ON MATCH  SET m.score=m.score+$sc,"
             "             m.visits=m.visits+1")
        with self.driver.session() as s:
            s.run(q, s=smiles, sc=score, ad=adv, re=regret, ep=epoch)

    def create_action(self, parent: str, child: str,
                      typ: str, params: str, delta: float):
        q = ("MATCH (p:Molecule {smiles:$p}),(c:Molecule {smiles:$c}) "
             "MERGE (p)-[r:ACTION {type:$t,params:$pa}]->(c) "
             "SET r.delta=$d")
        with self.driver.session() as s:
            s.run(q, p=parent, c=child, t=typ, pa=params, d=delta)

    # ---------- query ----------
    def top_k(self, k: int) -> List[Dict[str, Any]]:
        q = ("MATCH (m:Molecule) "
             "RETURN m.smiles AS s, m.score AS sc "
             "ORDER BY sc DESC LIMIT $k")
        with self.driver.session() as s:
            return [dict(r) for r in s.run(q, k=k)]

    # ---------- pruning mark ----------
    def mark_cold(self, smiles: str):
        with self.driver.session() as s:
            s.run("MATCH (m:Molecule {smiles:$s}) SET m.cold=true", s=smiles)