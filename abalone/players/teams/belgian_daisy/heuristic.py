"""Belgian Daisy tournament heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# Belgian Daisy: 80 total game moves, 10s per move.
BELGIAN_DAISY_WEIGHTS = {
    "marble": 50000.0,   # Material advantage must stay dominant.
    "center": 85.0,      # Useful positional signal, but no longer dominates tactical terms.
    "cohesion": 70.0,    # Keep groups connected without over-constraining movement.
    "cluster": 25.0,     # Cohesion does most of the group-shape work.
    "edge_pressure": 121.0,  # Combined edge safety and rim pressure remain important for depth-3 play.
    "formation": 88.0,   # Reward strong lines and wedges that deeper search can exploit.
    "push": 330.0,       # Restore meaningful offensive pressure for depth-3 play.
    "mobility": 38.0,    # More freedom helps avoid passive depth-3 lines.
    "stability": 52.0,   # Enough safety to avoid blunders without freezing the agent.
}

evaluate_belgian_daisy = build_weighted_evaluator(BELGIAN_DAISY_WEIGHTS)
