"""Belgian Daisy tournament agents."""

from ...types import AgentDefinition
from .heuristic import evaluate_belgian_daisy


def _build_belgian_agent(index: int) -> AgentDefinition:
    return AgentDefinition(
        id=f"tournament-belgian-{index}",
        label=f"Tournament: Belgian Daisy {index}",
        owner="Tournament",
        evaluator=evaluate_belgian_daisy,
        default_depth=7,
        tie_break="lexicographic",
        max_quiescence_depth=9,
    )


AGENTS = [_build_belgian_agent(index) for index in range(1, 4)]
