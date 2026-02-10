#!/usr/bin/env python3
"""Abalone game entry point."""

import sys
from .game import Game
from .board import Board, BLACK, WHITE
from .state_space import print_state_space_summary, generate_raw_moves, generate_legal_moves


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--state-space':
        # Just print state space analysis
        board = Board()
        board.setup_standard()
        print(board.display())

        for player in [BLACK, WHITE]:
            raw = generate_raw_moves(board, player)
            print_state_space_summary(board, player)

            # Show dedup stats
            unique_raw = set()
            for m in raw:
                key = (tuple(sorted(m.marbles)), m.direction)
                unique_raw.add(key)
            print(f"  Raw moves unique (still no legality): {len(unique_raw)}")
            print()
        return

    game = Game()
    game.play()


if __name__ == '__main__':
    main()
