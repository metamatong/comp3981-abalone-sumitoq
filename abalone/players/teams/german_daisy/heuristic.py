"""German Daisy tournament heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# German Daisy: 80 total game moves, 10s per move.
GERMAN_DAISY_WEIGHTS = {
    # old german weights
    # "marble": 41072.002228,
    # "center": 555.088303,
    # "cohesion": 110.633331,
    # "cluster": 32.72568,
    # "edge_pressure": 76.348758,
    # "formation": 126.781307,
    # "push": 111.278449,
    # "mobility": 18.202237,
    # "stability": 32.83995,
    # new german weights
    "center": 669.645522,
    "cluster": 38.700176,
    "cohesion": 130.051498,
    "edge_pressure": 89.426115,
    "formation": 146.246046,
    "marble": 41716.552397,
    "mobility": 14.614685,
    "push": 87.461575,
    "stability": 39.55633
}

evaluate_german_daisy = build_weighted_evaluator(GERMAN_DAISY_WEIGHTS)
