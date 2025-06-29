import json
import random
import os
from typing import List, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import AllChem

# ================= Unified Action Rulebook Loading ============================
# Load the single source of truth: rdkit_action_rulebook.json
RULEBOOK_PATH = os.path.join(os.path.dirname(__file__), 'rdkit_action_rulebook.json')
with open(RULEBOOK_PATH, 'r', encoding='utf-8') as f:
    ACTION_RULEBOOK = json.load(f)

# ================= RDKit Action Implementations (Unchanged) ===================
def swap_scaffold_precisely(parent_smiles, old_scaffold_smarts, new_scaffold_smiles, **kwargs):
    # ... (your exact swap logic here) ...
    print(f"Executing scaffold_swap: {parent_smiles} -> {new_scaffold_smiles}")
    # For the sake of example, we return a simulation result
    return new_scaffold_smiles

def add_functional_group_rdkit(parent_smiles, group_smiles, **kwargs):
    # ... (your existing logic for adding functional groups goes here) ...
    print(f"Executing add_functional_group: {parent_smiles} + {group_smiles}")
    return parent_smiles + "." + group_smiles

# ================= Action Sampling and Proposing ==============================
def sample(k: int = 10, parent_smiles: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Sample k actions from the rulebook. Optionally, parent_smiles can be used for future context-aware sampling.
    """
    all_actions = list(ACTION_RULEBOOK.items())
    if k > len(all_actions):
        k = len(all_actions)
    sampled = random.sample(all_actions, k)
    return [
        {"name": name, **info}
        for name, info in sampled
    ]

def propose_mixed_actions(parent_smiles: str, depth: int, k_init: int) -> List[Dict[str, Any]]:
    """
    Propose a mixture of actions for a given parent_smiles and search depth.
    This can be extended for more sophisticated strategies.
    """
    # For now, just sample k_init actions from the rulebook
    return sample(k=k_init, parent_smiles=parent_smiles)

# ================= Prepare Actions for LLM ====================================
def prepare_actions_for_llm() -> List[Dict[str, str]]:
    """
    Prepare a list of available actions for the LLM, extracting only the name and description
    from the unified ACTION_RULEBOOK.
    """
    llm_actions = []
    for action_name, action_info in ACTION_RULEBOOK.items():
        llm_actions.append({
            "name": action_name,
            "description": action_info.get("description", "")
        })
    return llm_actions

# ================= Action Dispatcher ==========================================
def execute_action(parent_smiles: str, action_name: str) -> Optional[str]:
    """
    Execute the specified action using the unified ACTION_RULEBOOK.
    - parent_smiles: The input molecule SMILES string.
    - action_name: The key in ACTION_RULEBOOK.
    """
    action_info = ACTION_RULEBOOK.get(action_name)
    if not action_info:
        print(f"Action '{action_name}' not found in rulebook.")
        return None

    action_type = action_info.get("type")
    params = action_info.get("params", {})

    # Dispatch to the correct RDKit implementation based on action type
    if action_type == "scaffold_swap":
        return swap_scaffold_precisely(parent_smiles, **params)
    elif action_type == "add_functional_group":
        return add_functional_group_rdkit(parent_smiles, **params)
    # Add more action types as needed
    else:
        print(f"Unknown action type: {action_type}")
        return None

# ================= Example Usage ==============================================
if __name__ == "__main__":
    print("--- Available actions for LLM ---")
    for action in prepare_actions_for_llm():
        print(f"- {action['name']}: {action['description']}")

    print("\n--- Execute new code (refactored) ---")
    chosen_action = "swap_to_furan"  # Example action name from rulebook
    result = execute_action("Nc1ccccc1", chosen_action)
    print(f"Result: {result}") 