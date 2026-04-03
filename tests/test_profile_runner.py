import contextlib
import importlib
import io
import re
import unittest

from abalone import native

if not native.is_available():
    raise unittest.SkipTest("native extension not built")


class ProfileRunnerTests(unittest.TestCase):
    def test_profile_main_emits_deterministic_summary_shape(self):
        run_module = importlib.import_module("run")
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            run_module._profile_main(
                ["--depth", "2", "--max-moves", "2", "--opening-seed", "7", "--top", "5"]
            )

        output = stdout.getvalue()
        self.assertRegex(output, r"TOTAL_SECONDS \d+\.\d+")
        self.assertRegex(output, r"TOTAL_CALLS \d+")
        self.assertIn("MOVES 2", output)
        self.assertIn("MOVE 1:", output)
        self.assertIn("MOVE 2:", output)
        self.assertIsNotNone(re.search(r"STATUS \{.*'game_over':", output))


if __name__ == "__main__":
    unittest.main()
