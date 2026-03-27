"""German Daisy tournament heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# German Daisy: 80 total game moves, 10s per move.
GERMAN_DAISY_WEIGHTS = {
    "marble": 40000.0,
    "center": 500.0,
    "cohesion": 100.0,
    "cluster": 30.0,
    "edge_pressure": 70.0,
    "formation": 120.0,
    "push": 120.0,
    "mobility": 20.0,
    "stability": 30.0,
}

evaluate_german_daisy = build_weighted_evaluator(GERMAN_DAISY_WEIGHTS)
