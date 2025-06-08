"""Meta-controller deciding between LLM agent and Graph GA."""
from typing import Callable, List


class MetaPolicy:
    """A very simple multi-armed bandit style policy."""

    def __init__(self, llm_agent_generate: Callable, ga_generate: Callable, window: int = 50):
        self.llm_agent_generate = llm_agent_generate
        self.ga_generate = ga_generate
        self.window = window
        self.history: List[float] = []

    def update(self, value: float) -> None:
        self.history.append(value)
        if len(self.history) > self.window:
            self.history.pop(0)

    def next_generator(self) -> Callable:
        """Select next generator based on moving average reward."""
        if not self.history:
            return self.llm_agent_generate
        avg = sum(self.history) / len(self.history)
        # Toy strategy: alternate depending on avg threshold
        if avg > 0.5:
            return self.llm_agent_generate
        return self.ga_generate
