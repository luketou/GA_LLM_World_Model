"""Wrapper for GB_GA_generate from guacamol_baselines."""
from typing import List, Callable

try:
    from guacamol_baselines.graph_ga.goal_directed_generation import GB_GA_generate
except Exception:  # pragma: no cover - fallback when package missing
    GB_GA_generate = None


def ga_generate(n: int, scoring_function: Callable[[str], float]) -> List[str]:
    """Generate ``n`` SMILES using the Graph GA baseline."""
    if GB_GA_generate is None:
        # TODO: handle package not available
        return ["C" for _ in range(n)]
    return GB_GA_generate(scoring_function, number_molecules=n)
