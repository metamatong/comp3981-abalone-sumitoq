"""File I/O helpers for state-space expansion and comparison.

This module wraps file reading, writing, and path resolution around the
pure-logic functions in :mod:`abalone.state_space`.
"""

from pathlib import Path
from typing import List, Optional, Tuple

from .state_space import (
    PositionListComparison,
    compare_position_list_lines,
    expand_position_list_text,
)


def export_position_list_states(input_path: str, output_path: Optional[str] = None) -> int:
    """Expand one compact state file and optionally write the serialized child states.

    :return: Number of child states generated.
    """
    child_states = expand_position_list_text(Path(input_path).read_text(encoding="utf-8"))
    if output_path:
        write_position_list_lines(child_states, output_path)
    else:
        print("\n".join(child_states))

    return len(child_states)


def expected_output_path_for_input(input_path: str) -> Path:
    """Resolve the conventional expected output path for a state-space input file."""
    source = Path(input_path)
    if source.parent.name != "state_space_inputs":
        raise ValueError(
            "Cannot infer expected output path. "
            "Input must be inside a `state_space_inputs/` directory."
        )

    output_dir = source.parent.parent / "state_space_outputs"
    return output_dir / f"{source.stem}.board"


def generated_output_path_for_expected(expected_path: str) -> Path:
    """Return a sibling path for storing generated output beside an expected `.board` file."""
    expected = Path(expected_path)
    return expected.with_name(f"{expected.stem}.generated{expected.suffix}")


def write_position_list_lines(lines: List[str], output_path: str) -> Path:
    """Write serialized child-state lines to disk with a trailing newline when non-empty."""
    resolved = Path(output_path)
    content = "\n".join(lines)
    resolved.write_text(f"{content}\n" if content else "", encoding="utf-8")
    return resolved


def compare_position_list_files(
    input_path: str,
    expected_path: Optional[str] = None,
) -> Tuple[PositionListComparison, Path]:
    """Generate child states from an input file and compare them to an expected file."""
    resolved_expected = Path(expected_path) if expected_path else expected_output_path_for_input(input_path)
    actual = expand_position_list_text(Path(input_path).read_text(encoding="utf-8"))
    expected = resolved_expected.read_text(encoding="utf-8").splitlines()
    return compare_position_list_lines(actual, expected), resolved_expected


def compare_and_save_position_list_files(
    input_path: str,
    expected_path: Optional[str] = None,
    generated_output_path: Optional[str] = None,
) -> Tuple[PositionListComparison, Path, Path]:
    """Compare generated child states to expected output and save the generated lines."""
    resolved_expected = Path(expected_path) if expected_path else expected_output_path_for_input(input_path)
    actual = expand_position_list_text(Path(input_path).read_text(encoding="utf-8"))
    expected = resolved_expected.read_text(encoding="utf-8").splitlines()
    resolved_generated = write_position_list_lines(
        actual,
        generated_output_path or str(generated_output_path_for_expected(str(resolved_expected))),
    )
    return compare_position_list_lines(actual, expected), resolved_expected, resolved_generated
