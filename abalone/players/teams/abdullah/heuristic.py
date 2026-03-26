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
    "push": 200.0,      # High priority on pushing opponent marbles

    # Combined edge safety and rim pressure
    "edge_pressure": 95.0,  # Balances avoiding exposed own marbles with pressuring rimmed opponent marbles

    # Moderate structural considerations
    "formation": 35.0,  # Moderate value for structured formations

    # Slightly increased mobility
    "mobility": 30.0,   # Encourages maintaining move options
}

evaluate_abdullah = build_weighted_evaluator(ABDULLAH_WEIGHTS)
