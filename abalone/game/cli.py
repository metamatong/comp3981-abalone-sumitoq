"""Abalone terminal game loop with optional AI players."""

from typing import Optional

from ..state_space import generate_legal_moves, print_state_space_summary
from .board import (
    BLACK,
    WHITE,
    DIRECTIONS,
    NAME_TO_DIR,
    Move,
    is_valid,
    neighbor,
    pos_to_str,
    str_to_pos,
)
from .config import CONTROLLER_AI, GameConfig, MODE_HVH
from .session import GameSession


class Game:
    def __init__(self, mode: str = MODE_HVH, human_side: int = BLACK, ai_depth: int = 2):
        config = GameConfig(mode=mode, human_side=human_side, ai_depth=ai_depth)
        self.session = GameSession(config=config)

    @property
    def board(self):
        return self.session.board

    @property
    def current_player(self):
        return self.session.current_player

    @property
    def move_history(self):
        return self.session.move_history

    def play(self):
        print("=== ABALONE ===")
        print("Type 'help' for commands.\n")

        while not self.is_game_over():
            print(self.board.display())
            pname = "Black(@)" if self.current_player == BLACK else "White(O)"
            controller = self.session.current_controller
            print(f"  Turn: {pname} [{controller.upper()}]")
            print(f"  Marbles: B={self.board.marble_count(BLACK)} W={self.board.marble_count(WHITE)}")
            print()

            if controller == CONTROLLER_AI:
                result = self.session.apply_agent_move()
                if "error" in result:
                    print(f"  AI error: {result['error']}")
                    break

                print(f"  >> {pname} (AI) plays {result['notation']}")
                if result["result"]["pushed"]:
                    print(f"     Pushed: {', '.join(result['result']['pushed'])}")
                if result["result"]["pushoff"]:
                    print("     PUSHED OFF THE BOARD!")
                search = result.get("search")
                if search:
                    print(
                        f"     Search: depth={search['depth']} nodes={search['nodes']} "
                        f"time={search['elapsed_ms']}ms"
                    )
                print()
                continue

            move = self._get_move()
            if move is None:
                continue

            payload = {
                "marbles": [pos_to_str(pos) for pos in move.marbles],
                "direction": [move.direction[0], move.direction[1]],
            }
            result = self.session.apply_human_move(payload)
            if "error" in result:
                print(f"  {result['error']}")
                continue

            print(f"\n  >> {pname} plays {result['notation']}")
            if result["result"]["pushed"]:
                print(f"     Pushed: {', '.join(result['result']['pushed'])}")
            if result["result"]["pushoff"]:
                print("     PUSHED OFF THE BOARD!")
            print()

        print(self.board.display())
        status = self.session.status()
        winner = status["winner"]
        if winner is None:
            print("  GAME OVER! No winner.")
            return

        wname = "Black(@)" if winner == BLACK else "White(O)"
        print(f"  GAME OVER! {wname} wins!")
        print(f"  Final score: Black {self.board.score[BLACK]} - {self.board.score[WHITE]} White")

    def is_game_over(self) -> bool:
        return self.session.status()["game_over"]

    def _get_move(self) -> Optional[Move]:
        try:
            raw = input("  Enter move> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit

        if not raw:
            return None

        if raw == "q":
            print("Goodbye!")
            raise SystemExit

        if raw == "help":
            self._print_help()
            return None

        if raw == "moves":
            self._show_moves()
            return None

        if raw == "state":
            print_state_space_summary(self.board, self.current_player)
            return None

        if raw == "undo":
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

    def _parse_move(self, text: str) -> Optional[Move]:
        try:
            if ">" in text:
                head, dir_name = text.split(">")
                dir_name = dir_name.upper()
                if dir_name not in NAME_TO_DIR:
                    return None

                direction = NAME_TO_DIR[dir_name]
                parts = head.split(":")
                count = int(parts[0])
                ends = parts[1].split("-")
                end1 = str_to_pos(ends[0])
                end2 = str_to_pos(ends[1])

                line_d = (end2[0] - end1[0], end2[1] - end1[1])
                steps = max(abs(line_d[0]), abs(line_d[1]))
                if steps == 0 or steps != count - 1:
                    return None

                unit = (line_d[0] // steps, line_d[1] // steps)
                marbles = tuple((end1[0] + i * unit[0], end1[1] + i * unit[1]) for i in range(count))
                return Move(marbles=marbles, direction=direction)

            parts = text.split(":")
            count = int(parts[0])
            pos_str = parts[1]
            start = str_to_pos(pos_str[:2])
            goal = str_to_pos(pos_str[2:4])

            dr = goal[0] - start[0]
            dc = goal[1] - start[1]
            if count == 0:
                return None
            if dr % count != 0 or dc % count != 0:
                return None

            direction = (dr // count, dc // count)
            if direction not in DIRECTIONS:
                return None

            marbles = tuple((start[0] + i * direction[0], start[1] + i * direction[1]) for i in range(count))
            return Move(marbles=marbles, direction=direction)
        except (ValueError, IndexError, KeyError):
            return None

    def _show_moves(self):
        legal = generate_legal_moves(self.board, self.current_player)
        if not legal:
            print("  No legal moves!")
            return

        singles = [move for move in legal if move.count == 1]
        doubles = [move for move in legal if move.count == 2]
        triples = [move for move in legal if move.count == 3]

        print(f"\n  Legal moves ({len(legal)} total):")
        if singles:
            print(f"  Single ({len(singles)}):")
            for move in singles:
                print(f"    {move.to_notation()}")

        if doubles:
            inline = [move for move in doubles if move.is_inline]
            broad = [move for move in doubles if not move.is_inline]
            if inline:
                print(f"  Double inline ({len(inline)}):")
                for move in inline:
                    print(f"    {move.to_notation(pushed=self._would_push(move))}")
            if broad:
                print(f"  Double broadside ({len(broad)}):")
                for move in broad:
                    print(f"    {move.to_notation()}")

        if triples:
            inline = [move for move in triples if move.is_inline]
            broad = [move for move in triples if not move.is_inline]
            if inline:
                print(f"  Triple inline ({len(inline)}):")
                for move in inline:
                    print(f"    {move.to_notation(pushed=self._would_push(move))}")
            if broad:
                print(f"  Triple broadside ({len(broad)}):")
                for move in broad:
                    print(f"    {move.to_notation()}")
        print()

    def _would_push(self, move: Move) -> bool:
        if not move.is_inline or move.count < 2:
            return False

        _, leading = move._leading_trailing()
        ahead = neighbor(leading, move.direction)
        opponent = WHITE if self.current_player == BLACK else BLACK
        return is_valid(ahead) and self.board.cells.get(ahead) == opponent

    def _undo(self):
        result = self.session.undo()
        if "error" in result:
            print(f"  {result['error']}")
            return
        print("  Undid last move.")

    def _print_help(self):
        print(
            """
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
"""
        )
