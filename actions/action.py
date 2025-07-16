"""High level action utilities using a unified registry."""
from typing import List, Dict, Any, Optional
from .registry import ActionRegistry

registry = ActionRegistry()

# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def sample(k: int = 10, parent_smiles: Optional[str] = None) -> List[Dict[str, Any]]:
    """Sample ``k`` actions from the registry."""
    return [a.__dict__ for a in registry.sample(k)]


def propose_mixed_actions(parent_smiles: str, depth: int, k_init: int) -> List[Dict[str, Any]]:
    """Simple mixed action proposal. Currently just sampling."""
    return sample(k=k_init, parent_smiles=parent_smiles)


def prepare_actions_for_llm() -> List[Dict[str, str]]:
    """Return actions formatted for LLM prompts."""
    return registry.list_for_llm()


def execute_action(parent_smiles: str, action_name: str) -> Optional[str]:
    """Execute an action by name via the registry."""
    return registry.execute(parent_smiles, action_name)


if __name__ == "__main__":
    print("--- Available actions for LLM ---")
    for action in prepare_actions_for_llm():
        print(f"- {action['name']}: {action['description']}")

    print("\n--- Execute example action ---")
    chosen_action = "swap_to_furan"
    result = execute_action("Nc1ccccc1", chosen_action)
    print(f"Result: {result}")
