"""Convenience wrapper for PPO training entry point.

Usage:
  python train_rl.py [args]

Implementation lives in: training/train_rl.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path so imports work consistently
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from training.train_rl import main

if __name__ == "__main__":
    main()
