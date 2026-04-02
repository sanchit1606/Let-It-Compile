"""CUPTI hardware counter collection via Nsight Compute CLI (ncu).

Phase 1 goal:
- Provide a Windows-friendly way to collect a small set of GPU performance
  counters (CUPTI-derived) that can be used as an RL state vector.

Notes:
- We intentionally use `ncu` as a subprocess rather than binding directly to the
  CUPTI C API; this is more reliable on Windows and avoids driver/toolkit ABI
  issues.
- On Windows/WDDM, counters often require Administrator privileges; when blocked
  Nsight Compute reports ERR_NVGPUCTRPERM.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from profiling.ncu_utils import (
    ERR_NVGPUCTRPERM,
    ncu_metric_smoke_test,
    resolve_ncu_path,
    run_ncu_command,
)


# Small, stable default metric set (mostly percentages).
# Values returned by ncu are in "percent of peak" or "%" unless stated.
DEFAULT_NCU_METRICS: Dict[str, str] = {
    # Occupancy proxy (most important for reg-cap exploration)
    "achieved_occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",

    # Memory behavior
    "l2_hit_rate": "lts__t_sector_hit_rate.pct",
    "dram_bw_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",

    # Compute utilization
    "sm_active_pct": "sm__active_cycles.avg.pct_of_peak_sustained_elapsed",

    # Optional: a divergence / utilization proxy (ratio; can be >1 in some cases)
    # Keep it out of the default RL vector unless you explicitly enable it.
    "warp_exec_efficiency_ratio": "smsp__thread_inst_executed_per_inst_executed.ratio",
}


@dataclass(frozen=True)
class CuptiCollectResult:
    ok: bool
    reason: str
    raw: Dict[str, float]
    normalized: Dict[str, float]
    stdout: str = ""
    stderr: str = ""


def _normalize_metric_value(key: str, value: float) -> float:
    """Normalize metric to roughly [0, 1] for RL state usage.

    - Most DEFAULT_NCU_METRICS are percentages -> divide by 100.
    - Ratio metrics are clamped to [0, 1] (best-effort).
    """

    if key.endswith("_ratio"):
        return float(max(0.0, min(1.0, value)))
    return float(max(0.0, min(1.0, value / 100.0)))


def _parse_ncu_csv(stdout: str) -> Dict[str, float]:
    """Parse `ncu --csv` output into a dict: metric_name -> float value.

    ncu sometimes prints preamble lines; the CSV portion typically starts at the
    header containing 'Metric Name'.
    """

    if not stdout:
        return {}

    # ncu prints a CSV table whose header line contains a "Metric Name" column.
    # Important: the header line often starts with many other columns ("ID",
    # "Process ID", ...). We must start parsing from the *beginning of the
    # header line*, not from the middle of the line where the substring occurs.
    metric_idx = stdout.find("Metric Name")
    if metric_idx == -1:
        return {}

    # Find the beginning of the line that contains the header.
    line_start = stdout.rfind("\n", 0, metric_idx)
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1

    csv_text = stdout[line_start:]
    out: Dict[str, float] = {}

    reader = csv.DictReader(io.StringIO(csv_text))
    for row in reader:
        name = (row.get("Metric Name") or row.get("MetricName") or "").strip()
        if not name:
            continue

        raw_val = (
            row.get("Metric Value")
            or row.get("Metric Val")
            or row.get("MetricValue")
            or row.get("Value")
            or ""
        ).strip()
        if not raw_val:
            continue

        raw_val = raw_val.replace(",", "")
        if raw_val.endswith("%"):
            raw_val = raw_val[:-1]

        try:
            out[name] = float(raw_val)
        except Exception:
            continue

    return out


def _metrics_arg(metric_names: Iterable[str]) -> str:
    # ncu expects a comma-separated metric list.
    return ",".join(metric_names)


class CUPTICollector:
    """Collect a small set of Nsight Compute metrics for a Python kernel runner."""

    def __init__(self, metrics: Optional[Dict[str, str]] = None):
        self.metrics = dict(metrics or DEFAULT_NCU_METRICS)

    def preflight(self) -> CuptiCollectResult:
        """Check whether counters are accessible (permission + tool availability)."""

        # Use an occupancy metric as a proxy that counters work.
        metric = self.metrics.get("achieved_occupancy", DEFAULT_NCU_METRICS["achieved_occupancy"])
        res = ncu_metric_smoke_test(metric=metric)
        if res.ok:
            return CuptiCollectResult(ok=True, reason="ok", raw={}, normalized={}, stdout=res.stdout, stderr=res.stderr)

        # Classify the most common failure modes.
        if res.reason == "ncu_not_found":
            return CuptiCollectResult(ok=False, reason="ncu_not_found", raw={}, normalized={}, stdout=res.stdout, stderr=res.stderr)
        if "permission_denied" in res.reason:
            return CuptiCollectResult(ok=False, reason=res.reason, raw={}, normalized={}, stdout=res.stdout, stderr=res.stderr)

        return CuptiCollectResult(ok=False, reason=res.reason, raw={}, normalized={}, stdout=res.stdout, stderr=res.stderr)

    def collect_from_python_file(
        self,
        script_path: str | Path,
        *,
        timeout_s: int = 120,
        python_exe: Optional[str] = None,
        ncu_path: str = "ncu",
    ) -> CuptiCollectResult:
        """Run a Python script under ncu and collect the configured metrics.

        The script should:
        - compile (JIT) whatever it needs,
        - run exactly one representative kernel launch,
        - synchronize the default CUDA stream.
        """

        script_path = Path(script_path)
        py_exe = python_exe or sys.executable

        # If counters are blocked, fail fast with a clear reason.
        pre = self.preflight()
        if not pre.ok:
            return pre

        metric_list = list(self.metrics.values())
        args = [
            "--metrics",
            _metrics_arg(metric_list),
            "--csv",
            "--target-processes",
            "all",
            py_exe,
            str(script_path),
        ]

        resolved_ncu = ncu_path
        if ncu_path == "ncu":
            resolved_ncu = resolve_ncu_path() or ""
        if not resolved_ncu:
            return CuptiCollectResult(ok=False, reason="ncu_not_found", raw={}, normalized={})

        try:
            res = run_ncu_command(resolved_ncu, args=args, timeout_s=timeout_s)
        except subprocess.TimeoutExpired as e:
            return CuptiCollectResult(
                ok=False,
                reason="timeout",
                raw={},
                normalized={},
                stdout=(getattr(e, "stdout", "") or ""),
                stderr=(getattr(e, "stderr", "") or ""),
            )
        except Exception as e:
            return CuptiCollectResult(
                ok=False,
                reason=f"spawn_error: {e}",
                raw={},
                normalized={},
            )

        stdout = res.stdout or ""
        stderr = res.stderr or ""
        combined = stdout + "\n" + stderr

        if ERR_NVGPUCTRPERM in combined:
            # Common on Windows if not admin.
            return CuptiCollectResult(ok=False, reason="permission_denied", raw={}, normalized={}, stdout=stdout, stderr=stderr)

        parsed = _parse_ncu_csv(combined)

        raw_out: Dict[str, float] = {}
        norm_out: Dict[str, float] = {}
        for key, metric_name in self.metrics.items():
            if metric_name not in parsed:
                continue
            v = float(parsed[metric_name])
            raw_out[key] = v
            norm_out[key] = _normalize_metric_value(key, v)

        ok = bool(raw_out)
        if ok:
            reason = "ok"
        else:
            reason = f"ncu_failed_rc_{res.returncode}" if res.returncode else "no_metrics_parsed"

        return CuptiCollectResult(
            ok=ok,
            reason=reason,
            raw=raw_out,
            normalized=norm_out,
            stdout=stdout,
            stderr=stderr,
        )

    def collect_from_python_code(
        self,
        code: str,
        *,
        timeout_s: int = 120,
        ncu_path: str = "ncu",
    ) -> CuptiCollectResult:
        """Convenience wrapper: write code to a temp file and profile it."""

        with tempfile.TemporaryDirectory(prefix="cupti_collect_") as td:
            script_path = Path(td) / "cupti_runner.py"
            script_path.write_text(code, encoding="utf-8")
            return self.collect_from_python_file(script_path, timeout_s=timeout_s, ncu_path=ncu_path)


def default_state_vector(order: Optional[Iterable[str]] = None) -> np.ndarray:
    """Return a zero-initialized default RL state vector for Phase 1.

    This is useful when profiling is unavailable; the environment can still run.

    By default we return the 4 main percentage metrics.
    """

    keys = list(order) if order is not None else [
        "achieved_occupancy",
        "l2_hit_rate",
        "dram_bw_pct",
        "sm_active_pct",
    ]
    return np.zeros(len(keys), dtype=np.float32)
