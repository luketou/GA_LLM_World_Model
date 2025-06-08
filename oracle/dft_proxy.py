"""Proxy for an expensive DFT oracle."""
import random


def evaluate(smiles: str) -> float:
    """Return a fake oracle score for the molecule."""
    # TODO: replace with real model or DFT call
    random.seed(hash(smiles) % (2**32))
    return random.random()
