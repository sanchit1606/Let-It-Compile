#!/usr/bin/env python3
"""One-time migration: rename existing result artifacts to include GPU tag.

This script is intentionally conservative:
- Only renames files when the destination doesn't already exist.
- Never deletes anything.
- Skips files that appear locked (Windows) without failing.

Typical usage:
  python scripts\\rename_results_gpu_tag.py --run-tag rtx3050_01 --gpu-tag rtx3050

If --gpu-tag is omitted, it's inferred as the part before the first '_' in --run-tag.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _safe_rename(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if dst.exists():
        return False
    try:
        src.rename(dst)
        return True
    except OSError:
        return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    results = repo_root / "results"
    logs = results / "logs"
    models = results / "models"

    p = argparse.ArgumentParser()
    p.add_argument("--run-tag", required=True, help="Run tag folder name, e.g. rtx3050_01")
    p.add_argument(
        "--gpu-tag",
        default=None,
        help="GPU tag to embed in filenames, e.g. rtx3050 (default: inferred from run-tag)",
    )
    args = p.parse_args()

    run_tag: str = args.run_tag
    gpu_tag: str = args.gpu_tag or run_tag.split("_", 1)[0]

    renamed = []
    skipped = []

    # 1) Per-run log dir artifacts
    run_log_dir = logs / run_tag
    if run_log_dir.exists():
        renamed.append(
            (run_log_dir / "training_summary.json", run_log_dir / f"training_summary_{gpu_tag}.json")
        )
        renamed.append((run_log_dir / "train_rl.log", run_log_dir / f"train_rl_{gpu_tag}.log"))
        renamed.append((run_log_dir / "evaluations.npz", run_log_dir / f"evaluations_{gpu_tag}.npz"))
        renamed.append((run_log_dir / "monitor.csv", run_log_dir / f"monitor_{gpu_tag}.csv"))
    else:
        skipped.append(f"Missing run log dir: {run_log_dir}")

    # 2) Root logs leftovers (older runs)
    renamed.append((logs / "train_rl.log", logs / f"train_rl_{gpu_tag}.log"))
    renamed.append((logs / "evaluations.npz", logs / f"evaluations_{gpu_tag}.npz"))

    # 3) TensorBoard artifacts
    tb_dir = logs / "tensorboard" / run_tag
    if tb_dir.exists():
        renamed.append((tb_dir / "progress.csv", tb_dir / f"progress_{gpu_tag}.csv"))
        # Event files: prefix with gpu_tag
        for ev in tb_dir.glob("events.out.tfevents.*"):
            dst = tb_dir / f"{gpu_tag}_{ev.name}"
            renamed.append((ev, dst))
    else:
        skipped.append(f"Missing tensorboard dir: {tb_dir}")

    # 4) Legacy best model path
    legacy_best = models / "best" / "best_model.zip"
    legacy_best_dst = models / "best" / f"best_model_{gpu_tag}.zip"
    renamed.append((legacy_best, legacy_best_dst))

    applied = 0
    for src, dst in renamed:
        if _safe_rename(src, dst):
            applied += 1
        else:
            # only report existing sources that couldn't be renamed
            if src.exists() and not dst.exists():
                skipped.append(f"Could not rename (locked?): {src} -> {dst}")

    print(f"[OK] GPU tag migration complete")
    print(f"  run_tag={run_tag}")
    print(f"  gpu_tag={gpu_tag}")
    print(f"  Renamed files: {applied}")
    if skipped:
        print("  Skipped:")
        for s in skipped:
            print(f"   - {s}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
