"""LangGraph based language model agent."""
from typing import List


class LangGraphAgent:
    """Simple stub for a LangGraph driven LLM agent."""

    def __init__(self):
        # TODO: integrate real LangGraph nodes and planning
        pass

    def generate(self, n: int, task_emb) -> List[str]:
        """Return ``n`` SMILES strings.``task_emb`` is unused in this stub."""
        # In a real system this would use LangGraph to plan and call an LLM.
        return ["C" for _ in range(n)]
