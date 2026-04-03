import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from abalone import native
from abalone.game.main import main
from abalone.game.board import (
    BLACK,
    WHITE,
    Board,
    DIRECTIONS,
    NEIGHBOR_TABLE,
    ORDERED_VALID_POSITIONS,
    is_valid,
    neighbor,
)
from abalone.file_handler import (
    compare_and_save_position_list_files,
    compare_position_list_files,
)
from abalone.state_space import (
    dump_position_list_state,
    generate_legal_moves,
    generate_move_notation_strings,
    generate_next_states,
    load_position_list_state,
)

if not native.is_available():
    raise unittest.SkipTest("native extension not built")

SAMPLE_INPUT = """b
C5b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
"""

SAMPLE_OUTPUT = """C6b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
B5b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
B4b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,G8b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G9w,H7w,H8w,H9w
B5b,C5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G5b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G7w,G8w,G9w,H5w,H7w,H8w,H9w
C5b,D5b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,H7b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H8w,H9w,I8w
C5b,D5b,E5b,E6b,E7b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E8w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E3b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E6b,E7b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E8w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
B5b,C5b,D5b,E4b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E3b,E4b,E6b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E6b,F5b,F6b,F7b,F8b,G5b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G7w,G8w,G9w,H5w,H7w,H8w,H9w
C4b,C5b,D5b,E4b,E5b,F5b,F6b,F7b,F8b,G6b,H6b,B3w,C3w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E3b,E4b,E5b,F5b,F6b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F6b,F7b,F8b,G6b,H6b,H7b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H8w,H9w,I8w
C5b,D3b,D5b,E4b,E5b,E6b,F6b,F7b,F8b,G6b,H6b,C2w,C3w,C4w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F5b,F7b,F8b,F9b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,D6b,E4b,E5b,E6b,F5b,F7b,F8b,G6b,H6b,C3w,C4w,C6w,D3w,D4w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F4b,F5b,F7b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F3w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F5b,F7b,F8b,G6b,H6b,I6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F5b,F6b,F8b,F9b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C4b,C5b,D5b,E4b,E5b,E6b,F5b,F6b,F8b,G6b,H6b,B3w,C3w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F4b,F5b,F6b,F8b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F3w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F9b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,E8b,F5b,F6b,F7b,G6b,H6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,D6b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,H6b,C3w,C4w,C6w,D3w,D4w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D3b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,H6b,C2w,C3w,C4w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,H6b,I6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,I7b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,H5b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
C5b,D5b,E4b,E5b,E6b,F5b,F6b,F7b,F8b,G6b,I6b,C3w,C4w,D3w,D4w,D6w,E7w,F4w,G5w,G7w,G8w,G9w,H7w,H8w,H9w
"""


class StateSpaceFileTests(unittest.TestCase):
    """Regression checks for compact board-file parsing and one-ply expansion."""

    def test_load_position_list_state_parses_turn_and_infers_score(self):
        board, player = load_position_list_state(SAMPLE_INPUT)

        self.assertEqual(player, BLACK)
        self.assertEqual(board.marble_count(BLACK), 11)
        self.assertEqual(board.marble_count(WHITE), 14)
        self.assertEqual(board.score[BLACK], 0)
        self.assertEqual(board.score[WHITE], 3)

    def test_generate_next_states_match_sample_state_set(self):
        board, player = load_position_list_state(SAMPLE_INPUT)

        actual = {dump_position_list_state(child) for child in generate_next_states(board, player)}
        expected = {line.strip() for line in SAMPLE_OUTPUT.splitlines() if line.strip()}

        self.assertEqual(len(actual), 32)
        self.assertEqual(actual, expected)

    def test_state_input_file_writes_child_states(self):
        expected = {line.strip() for line in SAMPLE_OUTPUT.splitlines() if line.strip()}

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "sample.input"
            output_path = Path(tmpdir) / "sample.board"
            input_path.write_text(SAMPLE_INPUT, encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()):
                main(
                    [
                        "--state-space",
                        "--state-input-file",
                        str(input_path),
                        "--state-output-file",
                        str(output_path),
                    ]
                )

            actual = {line.strip() for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()}
            self.assertEqual(actual, expected)

    def test_compare_position_list_files_reports_exact_order_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "state_space_inputs"
            output_dir = Path(tmpdir) / "state_space_outputs"
            input_dir.mkdir()
            output_dir.mkdir()

            input_path = input_dir / "Test1.input"
            expected_path = output_dir / "Test1.board"
            input_path.write_text(SAMPLE_INPUT, encoding="utf-8")
            expected_path.write_text(SAMPLE_OUTPUT, encoding="utf-8")

            comparison, resolved_expected = compare_position_list_files(str(input_path))

            self.assertEqual(resolved_expected, expected_path)
            self.assertEqual(comparison.actual_lines, 32)
            self.assertEqual(comparison.expected_lines, 32)
            self.assertTrue(comparison.exact_match)
            self.assertTrue(comparison.same_set)
            self.assertEqual(comparison.mismatch_count, 0)

    def test_state_verify_prints_exact_match_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "state_space_inputs"
            output_dir = Path(tmpdir) / "state_space_outputs"
            input_dir.mkdir()
            output_dir.mkdir()

            input_path = input_dir / "Test1.input"
            expected_path = output_dir / "Test1.board"
            input_path.write_text(SAMPLE_INPUT, encoding="utf-8")
            expected_path.write_text(SAMPLE_OUTPUT, encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                main(
                    [
                        "--state-space",
                        "--state-input-file",
                        str(input_path),
                        "--state-verify",
                    ]
                )

            output = stdout.getvalue()
            generated_path = output_dir / "Test1.generated.board"
            self.assertTrue(generated_path.exists())
            self.assertIn("exact_match=True", output)
            self.assertIn("same_set=True", output)
            self.assertIn("mismatch_count=0", output)
            self.assertIn(f"generated={generated_path}", output)

    def test_compare_and_save_position_list_files_writes_generated_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "state_space_inputs"
            output_dir = Path(tmpdir) / "state_space_outputs"
            input_dir.mkdir()
            output_dir.mkdir()

            input_path = input_dir / "Test1.input"
            expected_path = output_dir / "Test1.board"
            input_path.write_text(SAMPLE_INPUT, encoding="utf-8")
            expected_path.write_text(SAMPLE_OUTPUT, encoding="utf-8")

            comparison, resolved_expected, generated_path = compare_and_save_position_list_files(str(input_path))

            self.assertEqual(resolved_expected, expected_path)
            self.assertEqual(generated_path, output_dir / "Test1.generated.board")
            self.assertTrue(generated_path.exists())
            self.assertTrue(comparison.exact_match)

    # Verify that a single marble placed in the center of the board
    # produces six legal next states (one for each possible direction).
    def test_single_center_marble_has_six_next_states(self):
        sample_input = """b
    E5b
    """
        board, player = load_position_list_state(sample_input)
        actual = generate_next_states(board, player)

        self.assertEqual(player, BLACK)
        self.assertEqual(len(actual), 6)


    # Verify that the exact board states generated from a single
    # center marble match the expected neighboring positions.
    def test_single_center_marble_generates_expected_states(self):
        sample_input = """b
    E5b
    """
        expected = {
            "D4b",
            "D5b",
            "E4b",
            "E6b",
            "F5b",
            "F6b",
        }

        board, player = load_position_list_state(sample_input)
        actual = {dump_position_list_state(child) for child in generate_next_states(board, player)}

        self.assertEqual(actual, expected)


    # Verify that a marble placed on a board corner has fewer legal moves
    # due to board boundaries limiting available directions.
    def test_single_corner_marble_has_fewer_legal_moves(self):
        sample_input = """b
    A1b
    """
        board, player = load_position_list_state(sample_input)
        actual = generate_next_states(board, player)

        self.assertEqual(player, BLACK)
        self.assertEqual(len(actual), 3)

    def test_generated_moves_pass_full_and_fast_validators(self):
        board, player = load_position_list_state(SAMPLE_INPUT)

        for move in generate_legal_moves(board, player):
            self.assertTrue(board.is_generated_move_legal(move, player))
            self.assertTrue(board.is_legal_move(move, player))

    def test_generated_moves_pass_full_and_fast_validators_on_standard_board(self):
        board = Board()
        board.setup_standard()

        for player in (BLACK, WHITE):
            for move in generate_legal_moves(board, player):
                self.assertTrue(board.is_generated_move_legal(move, player))
                self.assertTrue(board.is_legal_move(move, player))

    def test_state_space_fixture_files_preserve_exact_move_and_state_order(self):
        fixture_root = Path("abalone")
        for input_path in sorted((fixture_root / "state_space_inputs").glob("Test*.input")):
            board, player = load_position_list_state(input_path.read_text(encoding="utf-8"))
            expected_states = (
                (fixture_root / "state_space_outputs" / f"{input_path.stem}.board")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            expected_moves = (
                (fixture_root / "state_space_outputs" / f"{input_path.stem}.move")
                .read_text(encoding="utf-8")
                .splitlines()
            )

            actual_moves = generate_move_notation_strings(board, player)
            actual_states = [dump_position_list_state(child) for child in generate_next_states(board, player)]

            self.assertEqual(actual_moves, expected_moves, msg=input_path.name)
            self.assertEqual(actual_states, expected_states, msg=input_path.name)

    def test_neighbor_table_matches_legacy_neighbor_and_is_valid_helpers(self):
        for pos in ORDERED_VALID_POSITIONS:
            for direction_index, direction in enumerate(DIRECTIONS):
                expected = neighbor(pos, direction)
                actual = NEIGHBOR_TABLE[pos][direction_index]
                if is_valid(expected):
                    self.assertEqual(actual, expected)
                else:
                    self.assertIsNone(actual)


if __name__ == "__main__":
    unittest.main()
