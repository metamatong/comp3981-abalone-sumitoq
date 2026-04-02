#!/usr/bin/env python3
"""
Run the Abalone game.

  python run.py          → web UI at http://localhost:8080
  python run.py cli      → terminal UI
  python run.py state    → state-space analysis
  python run.py match    → AI-vs-AI benchmark rounds
  python run.py duel     → AI-vs-AI duel reports (single game or one-vs-all gauntlet)
  python run.py profile  → deterministic search/profile benchmark
  python run.py state --state-input-file foo.input --state-output-file foo.board
                         → expand one input board file to all next legal states
                           produces foo.board (board states) and foo.move (move notations)
  python run.py state --state-input-file abalone/state_space_inputs/Test1.input --state-verify
                         → compare generated child states against expected output
"""
import argparse
import cProfile
import pstats
import sys


def _profile_main(argv):
    parser = argparse.ArgumentParser(description="Profile a deterministic AI-vs-AI search run.")
    parser.add_argument("--depth", type=int, default=4, help="Shared AI depth override.")
    parser.add_argument("--max-moves", type=int, default=4, help="Maximum game moves to play.")
    parser.add_argument("--opening-seed", type=int, default=7, help="Opening seed for deterministic black move.")
    parser.add_argument("--top", type=int, default=20, help="Number of hotspot rows to print.")
    args = parser.parse_args(argv)

    from abalone.game.config import GameConfig
    from abalone.game.session import GameSession

    session = GameSession(
        config=GameConfig(
            mode="ava",
            ai_depth=args.depth,
            black_ai_id="default",
            white_ai_id="default",
            max_moves=args.max_moves,
        ),
        opening_seed=args.opening_seed,
    )
    session.reset()

    profiler = cProfile.Profile()
    profiler.enable()
    while not session.status()["game_over"]:
        result = session.apply_agent_move()
        if "error" in result:
            raise RuntimeError(result["error"])
    profiler.disable()

    stats = pstats.Stats(profiler)
    print(f"TOTAL_SECONDS {stats.total_tt:.3f}")
    print(f"TOTAL_CALLS {stats.total_calls}")
    print(f"PRIMITIVE_CALLS {stats.prim_calls}")
    print(f"MOVES {len(session.move_history)}")
    print(f"STATUS {session.status()}")
    for index, entry in enumerate(session.move_history, start=1):
        search = entry.get("search") or {}
        print(
            f"MOVE {index}: {entry.get('agent_id')} {entry['move']} "
            f"elapsed_ms={search.get('elapsed_ms')} "
            f"nodes={search.get('nodes')} "
            f"completed_depth={search.get('completed_depth')} "
            f"source={search.get('decision_source')}"
        )
    stats.sort_stats("cumulative").print_stats(args.top)

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'web'

    if mode == 'cli':
        from abalone.game.main import main
        main(sys.argv[2:])
    elif mode == 'state':
        from abalone.game.main import main
        main(['--state-space'] + sys.argv[2:])
    elif mode == 'match':
        from abalone.game.match import main
        main(sys.argv[2:])
    elif mode == 'duel':
        from abalone.game.duel import main
        main(sys.argv[2:])
    elif mode == 'profile':
        _profile_main(sys.argv[2:])
    else:
        from abalone.game.server import run
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
        run(port)
