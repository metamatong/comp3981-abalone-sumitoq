"""Standard board tournament heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# Standard board: 10 min game clock, 5s per move, no move limit.
STANDARD_WEIGHTS = {
    "center": 444.070642,
    "cluster": 61.900624,
    "cohesion": 249.471523,
    "edge_pressure": 207.366499,
    "formation": 295.954056,
    "marble": 51858.270528,
    "mobility": 114.232688,
    "push": 1465.320381,
    "stability": 145.829084,
}

evaluate_standard = build_weighted_evaluator(STANDARD_WEIGHTS)
