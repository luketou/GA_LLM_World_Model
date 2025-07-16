from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import os

RULEBOOK_PATH = os.path.join(os.path.dirname(__file__), 'rdkit_action_rulebook.json')

@dataclass
class Action:
    name: str
    type: str
    description: str
    params: Dict[str, Any]

def load_rulebook(path: str = RULEBOOK_PATH) -> Dict[str, Action]:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    actions = {}
    for name, info in raw.items():
        actions[name] = Action(
            name=name,
            type=info.get('type', ''),
            description=info.get('description', ''),
            params=info.get('params', {})
        )
    return actions

class ActionRegistry:
    """Lightweight registry for available actions."""
    def __init__(self, rulebook_path: str = RULEBOOK_PATH):
        self.rulebook_path = rulebook_path
        self.actions: Dict[str, Action] = load_rulebook(rulebook_path)

    def list_for_llm(self) -> List[Dict[str, str]]:
        return [{"name": a.name, "description": a.description} for a in self.actions.values()]

    def get(self, name: str) -> Optional[Action]:
        return self.actions.get(name)

    def sample(self, k: int) -> List[Action]:
        import random
        if k > len(self.actions):
            k = len(self.actions)
        return random.sample(list(self.actions.values()), k)

    def execute(self, parent_smiles: str, action_name: str) -> Optional[str]:
        act = self.get(action_name)
        if not act:
            print(f"Action '{action_name}' not found in rulebook.")
            return None
        if act.type == 'scaffold_swap':
            return swap_scaffold_precisely(parent_smiles, **act.params)
        elif act.type == 'add_functional_group':
            return add_functional_group_rdkit(parent_smiles, **act.params)
        else:
            print(f"Unknown action type: {act.type}")
            return None

# RDKit specific implementations remain unchanged here

def swap_scaffold_precisely(parent_smiles, old_scaffold_smarts, new_scaffold_smiles, **kwargs):
    print(f"Executing scaffold_swap: {parent_smiles} -> {new_scaffold_smiles}")
    return new_scaffold_smiles


def add_functional_group_rdkit(parent_smiles, group_smiles, **kwargs):
    print(f"Executing add_functional_group: {parent_smiles} + {group_smiles}")
    return parent_smiles + '.' + group_smiles
