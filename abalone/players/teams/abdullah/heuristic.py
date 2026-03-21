"""Abdullah's heuristic configuration."""

from ....ai.heuristics import build_weighted_evaluator

ABDULLAH_WEIGHTS = {
    "marble": 50000.0,  # Primary objective: maximize material advantage

    # Aggressive playstyle
    # Reduced emphasis on positional safety
    "center": 25.0,     # Lower priority on central positioning
    "cohesion": 20.0,   # Less focus on tight grouping
    "cluster": 15.0,    # Reduced importance of large clusters
    "stability": 15.0,  # Lower concern for defensive stability

    # Strong offensive prioritization
    "push": 1000.0,      # High priority on pushing opponent marbles
    "threat": 150.0,    # Strong emphasis on creating push threats

    # Moderate defensive considerations
    "edge": 40.0,       # Some penalty for edge proximity
    "formation": 35.0,  # Moderate value for structured formations

    # Slightly increased mobility
    "mobility": 30.0,   # Encourages maintaining move options
}

evaluate_abdullah = build_weighted_evaluator(ABDULLAH_WEIGHTS)