"""
Abalone game loop with minimal text UI.

Move notation:
  Inline:    {count}:{trailing_pos}{goal_pos}    e.g. 3:h7e7  1:e5e6
  Broadside: {count}:{end1}-{end2}>{DIR}         e.g. 3:e5-e7>NW

  Trailing = marble farthest from movement direction
  Goal     = where leading marble ends up after the move
  * suffix = pushed an opponent marble

Commands during play:
  moves   - list all legal moves
  state   - show state space summary
  undo    - undo last move
  q       - quit
"""

from .board import (
    Board, Move, Position, Direction,
    DIRECTIONS, DIRECTION_NAMES, NAME_TO_DIR,
    VALID_POSITIONS,
    BLACK, WHITE, EMPTY,
    ROW_LETTERS,
    pos_to_str, str_to_pos, neighbor, is_valid, opposite_dir,
)
from .state_space import generate_legal_moves, print_state_space_summary
from typing import Optional, List


class Game:
    def __init__(self):
        self.board = Board()
        self.board.setup_standard()
        self.current_player = BLACK
        self.move_history: List[dict] = []  # [{move, result, board_snapshot}]

    def play(self):
        """Main game loop."""
        print("=== ABALONE ===")
        print("Type 'help' for commands.\n")

        while not self.is_game_over():
            print(self.board.display())
            pname = "Black(@)" if self.current_player == BLACK else "White(O)"
            print(f"  Turn: {pname}")
            print(f"  Marbles: B={self.board.marble_count(BLACK)} W={self.board.marble_count(WHITE)}")
            print()

            move = self._get_move()
            if move is None:
                continue

            # Save snapshot for undo
            snapshot = self.board.copy()
            result = self.board.apply_move(move, self.current_player)

            self.move_history.append({
                'move': move,
                'result': result,
                'snapshot': snapshot,
                'player': self.current_player,
            })

            # Announce
            notation = move.to_notation(pushed=bool(result['pushed']))
            print(f"\n  >> {pname} plays {notation}")
            if result['pushed']:
                print(f"     Pushed: {', '.join(result['pushed'])}")
            if result['pushoff']:
                print(f"     PUSHED OFF THE BOARD!")
            print()

            self.current_player = WHITE if self.current_player == BLACK else BLACK

        # Game over
        print(self.board.display())
        winner = BLACK if self.board.score[BLACK] >= 6 else WHITE
        wname = "Black(@)" if winner == BLACK else "White(O)"
        print(f"  GAME OVER! {wname} wins!")
        print(f"  Final score: Black {self.board.score[BLACK]} - {self.board.score[WHITE]} White")

    def is_game_over(self) -> bool:
        return self.board.score[BLACK] >= 6 or self.board.score[WHITE] >= 6

    def _get_move(self) -> Optional[Move]:
        try:
            raw = input("  Enter move> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            exit()

        if not raw:
            return None

        if raw == 'q':
            print("Goodbye!")
            exit()

        if raw == 'help':
            self._print_help()
            return None

        if raw == 'moves':
            self._show_moves()
            return None

        if raw == 'state':
            print_state_space_summary(self.board, self.current_player)
            return None

        if raw == 'undo':
            self._undo()
            return None

        move = self._parse_move(raw)
        if move is None:
            print("  Invalid move format. Type 'help' for syntax.")
            return None

        if not self.board.is_legal_move(move, self.current_player):
            print("  Illegal move!")
            return None

        return move

    def _parse_move(self, s: str) -> Optional[Move]:
        """
        Parse move notation.
        Inline:    3:h7e7  or  1:e5e6
        Broadside: 3:e5-e7>NW
        """
        try:
            # Broadside format: count:end1-end2>DIR
            if '>' in s:
                head, dir_name = s.split('>')
                dir_name = dir_name.upper()
                if dir_name not in NAME_TO_DIR:
                    return None
                d = NAME_TO_DIR[dir_name]

                parts = head.split(':')
                count = int(parts[0])
                ends = parts[1].split('-')
                end1 = str_to_pos(ends[0])
                end2 = str_to_pos(ends[1])

                # Build the line of marbles from end1 to end2
                line_d = (end2[0] - end1[0], end2[1] - end1[1])
                steps = max(abs(line_d[0]), abs(line_d[1]))
                if steps == 0 or steps != count - 1:
                    return None
                unit = (line_d[0] // steps, line_d[1] // steps)
                marbles = tuple(
                    (end1[0] + i * unit[0], end1[1] + i * unit[1])
                    for i in range(count)
                )
                return Move(marbles=marbles, direction=d)

            # Inline format: count:startgoal
            parts = s.split(':')
            count = int(parts[0])
            pos_str = parts[1]
            start = str_to_pos(pos_str[:2])
            goal = str_to_pos(pos_str[2:4])

            dr = goal[0] - start[0]
            dc = goal[1] - start[1]

            # For inline, distance from start (trailing) to goal equals count
            # Direction = (dr/count, dc/count)
            if count == 0:
                return None

            # Check if dr and dc are divisible by count
            if dr % count != 0 or dc % count != 0:
                return None

            d = (dr // count, dc // count)
            if d not in DIRECTIONS:
                return None

            # Build marbles: start, start+d, start+2d, ..., start+(count-1)*d
            marbles = tuple(
                (start[0] + i * d[0], start[1] + i * d[1])
                for i in range(count)
            )
            return Move(marbles=marbles, direction=d)

        except (ValueError, IndexError, KeyError):
            return None

    def _show_moves(self):
        legal = generate_legal_moves(self.board, self.current_player)
        if not legal:
            print("  No legal moves!")
            return

        # Categorize
        singles = [m for m in legal if m.count == 1]
        doubles = [m for m in legal if m.count == 2]
        triples = [m for m in legal if m.count == 3]

        print(f"\n  Legal moves ({len(legal)} total):")
        if singles:
            print(f"  Single ({len(singles)}):")
            for m in singles:
                print(f"    {m.to_notation()}")
        if doubles:
            inline_d = [m for m in doubles if m.is_inline]
            broad_d = [m for m in doubles if not m.is_inline]
            if inline_d:
                print(f"  Double inline ({len(inline_d)}):")
                for m in inline_d:
                    pushed = self._would_push(m)
                    print(f"    {m.to_notation(pushed=pushed)}")
            if broad_d:
                print(f"  Double broadside ({len(broad_d)}):")
                for m in broad_d:
                    print(f"    {m.to_notation()}")
        if triples:
            inline_t = [m for m in triples if m.is_inline]
            broad_t = [m for m in triples if not m.is_inline]
            if inline_t:
                print(f"  Triple inline ({len(inline_t)}):")
                for m in inline_t:
                    pushed = self._would_push(m)
                    print(f"    {m.to_notation(pushed=pushed)}")
            if broad_t:
                print(f"  Triple broadside ({len(broad_t)}):")
                for m in broad_t:
                    print(f"    {m.to_notation()}")
        print()

    def _would_push(self, move: Move) -> bool:
        """Check if an inline move would push an opponent marble."""
        if not move.is_inline or move.count < 2:
            return False
        d = move.direction
        _, leading = move._leading_trailing()
        ahead = neighbor(leading, d)
        opponent = WHITE if self.current_player == BLACK else BLACK
        return is_valid(ahead) and self.board.cells.get(ahead) == opponent

    def _undo(self):
        if not self.move_history:
            print("  Nothing to undo.")
            return
        entry = self.move_history.pop()
        self.board = entry['snapshot']
        self.current_player = entry['player']
        print(f"  Undid: {entry['move'].to_notation()}")

    def _print_help(self):
        print("""
  === Move notation ===
  Inline:    {count}:{trailing}{goal}     e.g. 3:h7e7  1:e5e6  2:b3b5
  Broadside: {count}:{end1}-{end2}>{DIR}  e.g. 3:e5-e7>NW  2:c3-c4>NW

  trailing = back marble (before move)
  goal     = where front marble ends up (after move)
  DIR      = E, W, NE, NW, SE, SW

  === Commands ===
  moves  - list all legal moves
  state  - state space analysis
  undo   - undo last move
  help   - show this help
  q      - quit
""")
