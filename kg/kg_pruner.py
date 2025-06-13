"""
Daily/off-line graph pruning.
"""
from .kg_store import KGStore


def prune(store: KGStore,
          score_thresh: float = 0.05,
          min_visits: int = 3):
    q = ("MATCH (m:Molecule) "
         "WHERE m.score < $th AND m.visits < $v AND coalesce(m.cold,false) "
         "DETACH DELETE m")
    with store.driver.session() as s:
        s.run(q, th=score_thresh, v=min_visits)