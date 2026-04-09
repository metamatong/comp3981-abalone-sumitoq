"""German Daisy tournament agent."""

from ...types import AgentDefinition
from .heuristic import evaluate_german_daisy

AGENTS = [
    AgentDefinition(
        id="tournament-german",
        label="Tournament: German Daisy",
        owner="Tournament",
        evaluator=evaluate_german_daisy,
        default_depth=7,
        tie_break="lexicographic",
        max_quiescence_depth=9,
        forced_finish_enabled=False,
    )
]
