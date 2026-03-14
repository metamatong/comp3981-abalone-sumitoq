"""Compatibility shim for shared AI type definitions."""

from ..ai.types import (
    AgentConfig,
    AgentDefinition,
    Evaluator,
    ResolvedAgentConfig,
    resolve_agent_config,
)

__all__ = [
    "AgentConfig",
    "AgentDefinition",
    "Evaluator",
    "ResolvedAgentConfig",
    "resolve_agent_config",
]
