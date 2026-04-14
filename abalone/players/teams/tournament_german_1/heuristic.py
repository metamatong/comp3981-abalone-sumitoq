"""Tournament German Daisy 1 heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# German Daisy: 80 total game moves, 10s per move.
TOURNAMENT_GERMAN_1_WEIGHTS = {
    "center": 669.645522,
    "cluster": 38.700176,
    "cohesion": 130.051498,
    "edge_pressure": 89.426115,
    "formation": 146.246046,
    "marble": 41716.552397,
    "mobility": 14.614685,
    "push": 87.461575,
    "stability": 39.55633,
}

evaluate_tournament_german_1 = build_weighted_evaluator(TOURNAMENT_GERMAN_1_WEIGHTS)
