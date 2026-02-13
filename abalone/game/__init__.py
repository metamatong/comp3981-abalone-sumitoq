"""Game runtime package for Abalone."""

from .config import (
    CONTROLLER_AI,
    CONTROLLER_HUMAN,
    MODE_AVA,
    MODE_HVA,
    MODE_HVH,
    GameConfig,
)

__all__ = [
    "GameConfig",
    "MODE_HVH",
    "MODE_HVA",
    "MODE_AVA",
    "CONTROLLER_HUMAN",
    "CONTROLLER_AI",
    "Game",
]


def __getattr__(name):
    if name == "Game":
        from .cli import Game

        return Game
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
