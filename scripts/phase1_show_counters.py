"""Phase 1 helper: collect real-kernel counters and write a CSV.

This is a convenience entrypoint for:
  experiments/phase1_collect_counters.py

Run this from an Administrator Command Prompt (recommended on Windows):
  cd /d "C:\\Users\\HP\\Desktop\\CD PROBLEM STATEMENT\\JIT Optimization across GPU stack"
  conda activate gpu-jit-opt
  python scripts\\phase1_show_counters.py

Output CSV:
  results/tables/phase1_result.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path (so `import experiments` works when running as:
# `python scripts\phase1_show_counters.py`)
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.phase1_collect_counters import run_phase1


if __name__ == "__main__":
    run_phase1()
