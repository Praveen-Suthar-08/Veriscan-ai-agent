"""Root evaluation harness entrypoint for VeriScan.

Delegates directly to veriscan.eval.run_eval.run_benchmark.
Usage:
    python eval/run_eval.py
"""

import sys
from pathlib import Path

# Ensure root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from veriscan.eval.run_eval import run_benchmark

if __name__ == "__main__":
    run_benchmark()
