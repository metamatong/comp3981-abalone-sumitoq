"""Game runtime package for Abalone.

Keep package-level exports lazy so importing submodules such as
``abalone.game.board`` does not eagerly import the rest of the runtime.
"""

__all__ = [
    "GameConfig",
    "MODE_HVH",
    "MODE_HVA",
    "MODE_AVA",
    "CONTROLLER_HUMAN",
    "CONTROLLER_AI",
    "Game",
]

_CONFIG_EXPORTS = {
    "GameConfig",
    "MODE_HVH",
    "MODE_HVA",
    "MODE_AVA",
    "CONTROLLER_HUMAN",
    "CONTROLLER_AI",
}


def __getattr__(name):
    """Resolve package exports lazily to avoid import-time cycles."""
    if name in _CONFIG_EXPORTS:
        from . import config

        return getattr(config, name)
    if name == "Game":
        from .cli import Game

        return Game
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
