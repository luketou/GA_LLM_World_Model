from typing import List, Dict
from .node import Node
from .uct import uct
from .progressive_widening import allow_expand
from actions.coarse_actions import sample as coarse_sample
from actions.fine_actions import expand as fine_expand
from kg.kg_store import KGStore

class MCTSEngine:
    def __init__(self, kg: KGStore, max_depth: int):
        self.kg = kg
        self.root = Node(smiles="ROOT", depth=0)
        self.best = self.root
        self.max_depth = max_depth
        self.epoch = 0

    # ----- external API -----
    def propose_actions(self, parent_smiles: str,
                        depth: int, k: int):
        if depth < 5:   # coarse 層
            return coarse_sample(parent_smiles, k)
        # PW 決定 unlock factor
        parent = self.root.children[parent_smiles]
        unlock = 2 * (parent.visits ** 0.6)
        return fine_expand(parent_smiles, unlock, top_k=k)

    def update_batch(self,
                     parent_smiles: str,
                     batch: List[str],
                     scores: List[float],
                     advantages: List[float]):
        parent = self.root.children.get(parent_smiles, self.root)
        baseline = sum(scores) / len(scores)
        # write
        for s, sc, adv in zip(batch, scores, advantages):
            n = parent.children.get(s) or Node(smiles=s, depth=parent.depth+1)
            n.visits += 1; n.total_score += sc
            n.advantage = adv; n.regret = baseline - sc
            parent.children[s] = n
            self.kg.create_molecule(s, sc, adv, n.regret, self.epoch)
            if sc > self.best.mean_score:
                self.best = n
        parent.visits += 1
        self.epoch += 1

    def select_child(self, parent_smiles: str):
        parent = self.root.children.get(parent_smiles, self.root)
        choices = parent.children.values()
        best_val, best = -1e9, None
        for ch in choices:
            val = uct(parent.visits, ch.visits,
                      ch.mean_score, ch.advantage)
            if val > best_val:
                best_val, best = val, ch
        return best