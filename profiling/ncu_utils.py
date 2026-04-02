"""Utilities for working with Nsight Compute (ncu).

Primary purpose: provide a reliable smoke test for CUPTI performance counter
permissions and surface actionable guidance when counters are blocked.

On Windows, collecting hardware metrics commonly requires running the command
prompt as Administrator; otherwise Nsight Compute emits `ERR_NVGPUCTRPERM`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


ERR_NVGPUCTRPERM = "ERR_NVGPUCTRPERM"


@dataclass(frozen=True)
class NcuSmokeTestResult:
    ok: bool
    reason: str
    stdout: str = ""
    stderr: str = ""
    ncu_path: Optional[str] = None


def _is_windows_admin() -> Optional[bool]:
    if os.name != "nt":
        return None
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return None


def _resolve_executable(name: str) -> Optional[str]:
    """Resolve an executable on PATH.

    Uses shutil.which first, then a Windows-only `where` fallback.
    """

    path = shutil.which(name)
    if path:
        return path

    if os.name == "nt":
        try:
            res = subprocess.run(
                ["where", name],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if res.returncode == 0:
                first = (res.stdout or "").strip().splitlines()[0].strip()
                return first or None
        except Exception:
            return None

    return None


def _run_ncu_command(ncu_path: str, args: list[str], timeout_s: int) -> subprocess.CompletedProcess:
    """Run ncu robustly across Windows .cmd/.bat shims."""

    p = Path(ncu_path)
    lower = str(p).lower()

    # Prefer adjacent .exe when a shim exists.
    if os.name == "nt" and p.suffix.lower() in {".cmd", ".bat"}:
        exe_candidate = p.with_suffix(".exe")
        if exe_candidate.exists():
            ncu_path = str(exe_candidate)
            lower = ncu_path.lower()

    if os.name == "nt" and (lower.endswith(".cmd") or lower.endswith(".bat")):
        # Execute batch shims via cmd.exe (shell=True) so quoting is interpreted
        # the same way as an interactive Command Prompt.
        cmdline = subprocess.list2cmdline([ncu_path, *args])
        return subprocess.run(
            cmdline,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            shell=True,
        )

    return subprocess.run(
        [ncu_path, *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def ncu_metric_smoke_test(
    metric: str = "sm__warps_active.avg.pct_of_peak_sustained_active",
    timeout_s: int = 60,
) -> NcuSmokeTestResult:
    """Run a minimal Numba kernel under `ncu --metrics <metric>`.

    Returns a structured result with stdout/stderr so callers can print friendly
    messages.
    """

    ncu_path = _resolve_executable("ncu")
    if not ncu_path:
        return NcuSmokeTestResult(ok=False, reason="ncu_not_found")

    # Write a tiny python file to avoid shell quoting pitfalls.
    code = """
import numpy as np
from numba import cuda

@cuda.jit
def add(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

n = 1024
x = cuda.to_device(np.ones(n, dtype=np.float32))
y = cuda.to_device(np.ones(n, dtype=np.float32))
z = cuda.device_array(n, dtype=np.float32)
add[32, 32](x, y, z)
cuda.default_stream().synchronize()
""".lstrip()

    with tempfile.TemporaryDirectory(prefix="ncu_smoke_") as td:
        script_path = Path(td) / "ncu_smoke_test.py"
        script_path.write_text(code, encoding="utf-8")

        # Use current python executable to ensure we test the active env.
        py_exe = sys.executable
        args = ["--metrics", metric, py_exe, str(script_path)]

        try:
            res = _run_ncu_command(ncu_path, args, timeout_s=timeout_s)
        except subprocess.TimeoutExpired as e:
            return NcuSmokeTestResult(
                ok=False,
                reason="timeout",
                stdout=(e.stdout or ""),
                stderr=(e.stderr or ""),
                ncu_path=ncu_path,
            )
        except Exception as e:
            return NcuSmokeTestResult(ok=False, reason=f"spawn_error: {e}", ncu_path=ncu_path)

    stdout = res.stdout or ""
    stderr = res.stderr or ""
    combined = (stdout + "\n" + stderr)

    if ERR_NVGPUCTRPERM in combined or "Performance Counters" in combined and "permission" in combined.lower():
        admin = _is_windows_admin()
        if os.name == "nt" and admin is False:
            reason = "permission_denied_not_admin"
        else:
            reason = "permission_denied"
        return NcuSmokeTestResult(
            ok=False,
            reason=reason,
            stdout=stdout,
            stderr=stderr,
            ncu_path=ncu_path,
        )

    # Consider it OK if the metric name appears in the report.
    if metric in combined:
        return NcuSmokeTestResult(ok=True, reason="ok", stdout=stdout, stderr=stderr, ncu_path=ncu_path)

    return NcuSmokeTestResult(
        ok=False,
        reason=f"ncu_failed_rc_{res.returncode}",
        stdout=stdout,
        stderr=stderr,
        ncu_path=ncu_path,
    )
