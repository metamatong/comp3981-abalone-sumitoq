"""Belgian Daisy tournament heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# Belgian Daisy: 80 total game moves, 10s per move.
BELGIAN_DAISY_WEIGHTS = {
    # belgian weights
    # "marble": 49635.333,   # Material advantage must stay dominant.
    # "center": 107.150,      # Useful positional signal, but no longer dominates tactical terms.
    # "cohesion": 85.425,    # Keep groups connected without over-constraining movement.
    # "cluster": 30.164,     # Cohesion does most of the group-shape work.
    # "edge_pressure": 125.891,  # Combined edge safety and rim pressure remain important for depth-3 play.
    # "formation": 103.780,   # Reward strong lines and wedges that deeper search can exploit.
    # "push": 418.161,       # Restore meaningful offensive pressure for depth-3 play.
    # "mobility": 30.245,    # More freedom helps avoid passive depth-3 lines.
    # "stability": 63.321,   # Enough safety to avoid blunders without freezing the agent.
    # german weights
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

evaluate_belgian_daisy = build_weighted_evaluator(BELGIAN_DAISY_WEIGHTS)
