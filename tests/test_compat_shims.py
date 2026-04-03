import importlib
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock


class CompatibilityShimTests(unittest.TestCase):
    def test_duel_main_preflights_missing_native_extension(self):
        duel = importlib.import_module("abalone.game.duel")

        with mock.patch("abalone.game.duel.native.is_available", return_value=False), mock.patch.object(
            duel,
            "_run_all_opponents_games",
        ) as run_games, self.assertRaises(SystemExit) as exc_info:
            duel.main(["--agent", "default", "--all-opponents"])

        self.assertIn("Native engine not built for this branch.", str(exc_info.exception))
        self.assertIn("Build it first:", str(exc_info.exception))
        run_games.assert_not_called()

    def test_native_bridge_and_state_space_import_in_clean_process(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = "\n".join(
            [
                "import sys",
                f"sys.path.insert(0, {repo_root.as_posix()!r})",
                "import abalone.native",
                "import abalone.state_space",
                "import abalone.ai.heuristics",
                "import abalone.ai.minimax",
            ]
        )

        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)

    def test_board_shim_reexports_core_board_symbols(self):
        shim = importlib.import_module("abalone.board")
        core = importlib.import_module("abalone.game.board")

        self.assertIs(shim.Board, core.Board)
        self.assertIs(shim.Move, core.Move)
        self.assertEqual(shim.BLACK, core.BLACK)
        self.assertEqual(shim.WHITE, core.WHITE)

    def test_main_and_server_shims_reexport_entrypoints(self):
        main_shim = importlib.import_module("abalone.main")
        main_core = importlib.import_module("abalone.game.main")
        server_shim = importlib.import_module("abalone.server")
        server_core = importlib.import_module("abalone.game.server")

        self.assertIs(main_shim.main, main_core.main)
        self.assertIs(server_shim.run, server_core.run)

    def test_player_shims_reexport_ai_modules(self):
        agent_shim = importlib.import_module("abalone.players.agent")
        agent_core = importlib.import_module("abalone.ai.agent")
        heuristics_shim = importlib.import_module("abalone.players.heuristics")
        heuristics_core = importlib.import_module("abalone.ai.heuristics")
        minimax_shim = importlib.import_module("abalone.players.minimax")
        minimax_core = importlib.import_module("abalone.ai.minimax")
        types_shim = importlib.import_module("abalone.players.types")
        types_core = importlib.import_module("abalone.ai.types")

        self.assertIs(agent_shim.choose_move, agent_core.choose_move)
        self.assertIs(agent_shim.choose_move_with_info, agent_core.choose_move_with_info)
        self.assertIs(minimax_shim.search_best_move, minimax_core.search_best_move)
        self.assertIs(heuristics_shim.evaluate_board, heuristics_core.evaluate_board)
        self.assertIs(types_shim.AgentConfig, types_core.AgentConfig)
        self.assertIs(types_shim.AgentDefinition, types_core.AgentDefinition)


if __name__ == "__main__":
    unittest.main()
