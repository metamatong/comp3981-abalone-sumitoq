"""Tournament Belgian Daisy 3 heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# Belgian Daisy: 80 total game moves, 10s per move.
TOURNAMENT_BELGIAN_3_WEIGHTS = {
    "marble": 41072.002228,
    "center": 555.088303,
    "cohesion": 110.633331,
    "cluster": 32.72568,
    "edge_pressure": 76.348758,
    "formation": 126.781307,
    "push": 111.278449,
    "mobility": 18.202237,
    "stability": 32.83995,
}

evaluate_tournament_belgian_3 = build_weighted_evaluator(TOURNAMENT_BELGIAN_3_WEIGHTS)
