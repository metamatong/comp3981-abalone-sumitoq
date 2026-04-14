"""Tournament German Daisy 3 heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# German Daisy: 80 total game moves, 10s per move.
TOURNAMENT_GERMAN_3_WEIGHTS = {
    "center": 699.233,
    "cluster": 38.211,
    "cohesion": 132.311,
    "edge_pressure": 95.931,
    "formation": 155.230,
    "marble": 42225.031,
    "mobility": 15.263,
    "push": 87.055,
    "stability": 39.914,
}

evaluate_tournament_german_3 = build_weighted_evaluator(TOURNAMENT_GERMAN_3_WEIGHTS)
