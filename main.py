"""Copyright (c) 2026 vecnode"""

import os
import sys

_SRC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

print("[main.py] Starting runtime.", flush=True)

from cli import parse_disco_argv
from run import main

if __name__ == "__main__":
    main(cli_overrides=parse_disco_argv(sys.argv[1:]))
