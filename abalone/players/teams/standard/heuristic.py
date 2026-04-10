"""Standard board tournament heuristic."""

from ....ai.heuristics import build_weighted_evaluator

# Standard board: 10 min game clock, 5s per move, no move limit.
STANDARD_WEIGHTS = {
    "marble": 50385.491,   # Material advantage must stay dominant.
    "center": 366.273,      # Useful positional signal, but no longer dominates tactical terms.
    "cohesion": 225.494,    # Keep groups connected without over-constraining movement.
    "cluster": 56.745,     # Cohesion does most of the group-shape work.
    "edge_pressure": 190.123,  # Combined edge safety and rim pressure remain important for depth-3 play.
    "formation": 280.124,   # Reward strong lines and wedges that deeper search can exploit.
    "push": 1375.517,       # Restore meaningful offensive pressure for depth-3 play.
    "mobility": 125.515,    # More freedom helps avoid passive depth-3 lines.
    "stability": 133.218,   # Enough safety to avoid blunders without freezing the agent.
}

evaluate_standard = build_weighted_evaluator(STANDARD_WEIGHTS)
