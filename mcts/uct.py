import math
def uct(parent_n: int, child_n: int,
        q: float, adv: float,
        c: float = 1.4, gamma: float = 0.3) -> float:
    explore = c * math.sqrt(math.log(parent_n + 1e-9) / (child_n + 1e-9))
    return q + explore + gamma * adv