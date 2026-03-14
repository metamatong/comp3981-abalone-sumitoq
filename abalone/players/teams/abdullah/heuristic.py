"""Abdullah's heuristic configuration."""

from ....ai.heuristics import build_weighted_evaluator

# Starts from the shared baseline. Abdullah can tune only this file later.
ABDULLAH_WEIGHTS = {
    "marble": 50000.0,
    "center": 40.0,
    "cohesion": 35.0,
    "cluster": 30.0,
    "edge": 60.0,
    "formation": 50.0,
    "push": 120.0,
    "threat": 80.0,
    "mobility": 20.0,
    "stability": 25.0,
}

evaluate_abdullah = build_weighted_evaluator(ABDULLAH_WEIGHTS)
