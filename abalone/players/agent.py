"""Public interface for choosing bot moves."""

from typing import Optional

from .minimax import SearchResult, search_best_move
from .types import AgentConfig


def choose_move(board, player: int, config: Optional[AgentConfig] = None):
    """Return the selected move for `player` using configured minimax search."""
    result = search_best_move(board, player, config=config)
    return result.move


def choose_move_with_info(board, player: int, config: Optional[AgentConfig] = None) -> SearchResult:
    """Return full search metadata, including selected move and diagnostics."""
    return search_best_move(board, player, config=config)
