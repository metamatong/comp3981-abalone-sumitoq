"""Standard board tournament heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# Standard board: 10 min game clock, 5s per move, no move limit.
STANDARD_WEIGHTS = {
    "center": 539.650629,
    "cluster": 73.054022,
    "cohesion": 297.928703,
    "edge_pressure": 259.036947,
    "formation": 348.401769,
    "marble": 52930.178228,
    "mobility": 93.707985,
    "push": 1767.763631,
    "stability": 187.479808
}

evaluate_standard = build_weighted_evaluator(STANDARD_WEIGHTS)
