"""Shared AI/search infrastructure for Abalone.

The package exports stay lazy so low-level modules can import AI helpers
without recursively importing the full search stack at package import time.
"""

__all__ = [
    "AgentConfig",
    "AgentDefinition",
    "DEFAULT_AGENT",
    "choose_move",
    "choose_move_with_info",
]

_TYPES_EXPORTS = {"AgentConfig", "AgentDefinition"}
_AGENT_EXPORTS = {"choose_move", "choose_move_with_info"}


def __getattr__(name):
    """Resolve top-level AI exports lazily to avoid import cycles."""
    if name in _TYPES_EXPORTS:
        from .types import AgentConfig, AgentDefinition

        exports = {
            "AgentConfig": AgentConfig,
            "AgentDefinition": AgentDefinition,
        }
        return exports[name]
    if name == "DEFAULT_AGENT":
        from .defaults import DEFAULT_AGENT

        return DEFAULT_AGENT
    if name in _AGENT_EXPORTS:
        from .agent import choose_move, choose_move_with_info

        exports = {
            "choose_move": choose_move,
            "choose_move_with_info": choose_move_with_info,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
