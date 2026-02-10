#!/usr/bin/env python3
"""
Run the Abalone game.

  python run.py          → web UI at http://localhost:8080
  python run.py cli      → terminal UI
  python run.py state    → state-space analysis
"""
import sys

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'web'

    if mode == 'cli':
        from abalone.main import main
        main()
    elif mode == 'state':
        sys.argv[1] = '--state-space'
        from abalone.main import main
        main()
    else:
        from abalone.server import run
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
        run(port)
