"""
Daily/off-line graph pruning.
"""
from .kg_store import KGStore


def prune(store: KGStore,
          score_thresh: float = 0.05,
          min_visits: int = 3):
    """離線／定期剪枝：刪除分數與訪問次數低、且已標記為冷分支的分子節點"""
    q = ("MATCH (m:Molecule) "
         "WHERE m.score < $th AND m.visits < $v AND coalesce(m.cold,false) = true "
         "DETACH DELETE m")
    with store.driver.session(database=store.database) as s:
        result = s.run(q, th=score_thresh, v=min_visits)
        return result.summary().counters.nodes_deleted