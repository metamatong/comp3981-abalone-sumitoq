"""Compatibility shim for CLI entry point."""

from .game.main import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
