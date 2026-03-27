"""Static registry of selectable AI agents."""

from typing import Dict, List

from ..ai.defaults import DEFAULT_AGENT
from ..ai.types import AgentDefinition
from .teams.abdullah import AGENTS as ABDULLAH_AGENTS
from .teams.cole import AGENTS as COLE_AGENTS
from .teams.jonah import AGENTS as JONAH_AGENTS
from .teams.kyle import AGENTS as KYLE_AGENTS
from .teams.belgian_daisy import AGENTS as BELGIAN_AGENTS
from .teams.german_daisy import AGENTS as GERMAN_AGENTS
from .teams.standard import AGENTS as STANDARD_AGENTS

ALL_AGENTS: List[AgentDefinition] = [
    # Comment out any you dont want to use when testing in terminal
    DEFAULT_AGENT,
    *KYLE_AGENTS,
    *ABDULLAH_AGENTS,
    *COLE_AGENTS,
    *JONAH_AGENTS,
    *BELGIAN_AGENTS,
    *GERMAN_AGENTS,
    *STANDARD_AGENTS,
]

AGENTS_BY_ID: Dict[str, AgentDefinition] = {agent.id: agent for agent in ALL_AGENTS}
AGENTS_BY_ID["baseline-balanced"] = DEFAULT_AGENT

DEFAULT_BLACK_AI_ID = "default"
DEFAULT_WHITE_AI_ID = "default"

def list_agents() -> List[AgentDefinition]:
    """Return every selectable agent in stable registry order."""
    return list(ALL_AGENTS)


def list_agent_metadata() -> List[dict]:
    """Return serializable agent metadata for CLI/web selectors."""
    return [
        {
            "id": agent.id,
            "label": agent.label,
            "owner": agent.owner,
            "default_depth": agent.default_depth,
            "tie_break": agent.tie_break,
        }
        for agent in ALL_AGENTS
    ]


def get_agent(agent_id: str) -> AgentDefinition:
    """Resolve an agent ID or raise a helpful error."""
    try:
        return AGENTS_BY_ID[agent_id]
    except KeyError as exc:
        raise ValueError(f"Unknown AI agent '{agent_id}'.") from exc
