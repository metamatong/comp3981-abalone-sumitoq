"""Compatibility shim for the shared AI move-selection interface."""

from ..ai.agent import choose_move, choose_move_with_info
from ..ai.minimax import SearchResult

__all__ = ["SearchResult", "choose_move", "choose_move_with_info"]
