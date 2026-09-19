"""Root data generation entrypoint for VeriScan.

Delegates directly to veriscan.data.generate_mock.
Usage:
    python data/generate_mock.py
"""

import sys
from pathlib import Path

# Ensure root is in python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from veriscan.data.generate_mock import (
    generate_all_datasets, SAMPLES_DIR, DEV_DIR, HELDOUT_DIR, WATERMARK_TEXT
)

if __name__ == "__main__":
    generate_all_datasets()
