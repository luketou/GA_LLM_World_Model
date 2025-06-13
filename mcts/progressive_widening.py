def allow_expand(parent_visits: int,
                 child_count: int,
                 alpha: float = 2,
                 beta: float = 0.6) -> bool:
    return child_count < alpha * (parent_visits ** beta)