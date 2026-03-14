"""Cole's heuristic configuration."""

from ....ai.heuristics import build_weighted_evaluator

# Starts from the shared baseline. Cole can tune only this file later.
COLE_WEIGHTS = {
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

evaluate_cole = build_weighted_evaluator(COLE_WEIGHTS)
