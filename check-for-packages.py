#!/usr/bin/env python3
"""
Package verification script for GPU JIT Optimization project.
Checks all required dependencies and prints their versions.
"""

import os
import sys
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("GPU JIT Optimization — Package Version Check")
print("=" * 70)
print()

packages = {
    "Core Deep Learning": [
        ("torch", "PyTorch"),
        ("torchvision", "Torchvision"),
        ("torchaudio", "Torchaudio"),
    ],
    "CUDA & GPU Control": [
        ("numba", "Numba (CUDA JIT)"),
        ("pynvml", "NVIDIA Management Library"),
        ("cuda", "CUDA Python Bindings"),
        ("nvtx", "NVIDIA Tools Extension"),
    ],
    "Reinforcement Learning": [
        ("stable_baselines3", "Stable-Baselines3 (PPO)"),
        ("gymnasium", "Gymnasium (RL environment)"),
    ],
    "Graph Neural Networks": [
        ("torch_geometric", "PyTorch Geometric (GNN)"),
        ("torch_scatter", "PyTorch Scatter"),
        ("torch_sparse", "PyTorch Sparse"),
    ],
    "Data & Visualization": [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("sklearn", "Scikit-learn"),
    ],
    "Utilities": [
        ("tqdm", "TQDM (Progress bars)"),
        ("rich", "Rich (Console formatting)"),
        ("pytest", "Pytest (Testing)"),
        ("black", "Black (Code formatter)"),
        ("isort", "isort (Import sorter)"),
        ("jupyter", "Jupyter"),
        ("ipykernel", "IPython Kernel"),
    ],
}

# Track results
results = {
    "installed": [],
    "missing": [],
    "errors": [],
}

for category, pkg_list in packages.items():
    print(f"[{category}]")
    for module_name, display_name in pkg_list:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            status = "✓"
            print(f"  {status} {display_name:<40} {version}")
            results["installed"].append((display_name, version))
        except ImportError as e:
            status = "✗"
            print(f"  {status} {display_name:<40} NOT INSTALLED")
            results["missing"].append(display_name)
        except Exception as e:
            status = "⚠"
            print(f"  {status} {display_name:<40} ERROR: {str(e)[:30]}")
            results["errors"].append((display_name, str(e)))
    print()

# Special checks
print("[Special Checks]")


def _resolve_executable(name: str) -> str | None:
    """Resolve an executable on PATH.

    On Windows, `shutil.which()` may return a `.cmd`/`.bat` shim. That's still
    runnable from a terminal (cmd/PowerShell), but not directly via
    `subprocess.run([...], shell=False)`. We handle that when executing.
    """

    path = shutil.which(name)
    if path:
        return path

    # Extra Windows-only fallback: `where.exe` sometimes finds tools even when
    # the current Python process has a different PATH view (e.g., VS Code).
    if os.name == "nt":
        try:
            import subprocess

            where_res = subprocess.run(
                ["where", name],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if where_res.returncode == 0:
                first = where_res.stdout.strip().splitlines()[0].strip()
                return first or None
        except Exception:
            return None

    return None


def _run_tool_version(tool: str, timeout_s: int = 5) -> tuple[bool, str, str | None]:
    """Run `<tool> --version` and return (ok, first_line, resolved_path)."""

    import subprocess

    resolved = _resolve_executable(tool)
    if not resolved:
        return False, "", None

    # If we resolved to a .bat/.cmd shim but an adjacent .exe exists, prefer it.
    # This avoids cmd.exe quoting edge cases and is the most reliable on Windows.
    if os.name == "nt":
        p = Path(resolved)
        if p.suffix.lower() in {".bat", ".cmd"}:
            exe_candidate = p.with_suffix(".exe")
            if exe_candidate.exists():
                resolved = str(exe_candidate)

    # If resolved is a cmd/bat shim, execute via cmd.exe.
    lower = resolved.lower()
    if os.name == "nt" and (lower.endswith(".cmd") or lower.endswith(".bat")):
        # Required cmd.exe quoting pattern when the command path is quoted:
        #   cmd.exe /c ""C:\Path With Spaces\tool.bat" --version"
        cmdline = f'""{resolved}" --version"'
        res = subprocess.run(
            ["cmd.exe", "/d", "/s", "/c", cmdline],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    else:
        res = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )

    text_out = (res.stdout or res.stderr or "").strip()
    first_line = text_out.splitlines()[0] if text_out else ""

    if res.returncode != 0:
        # Last-resort Windows fallback: let cmd.exe resolve PATHEXT/shims.
        # This mirrors interactive terminal behavior where `ncu` may be a .BAT.
        if os.name == "nt":
            try:
                res2 = subprocess.run(
                    f"{tool} --version",
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    shell=True,  # Uses cmd.exe /c on Windows
                )
                text2 = (res2.stdout or res2.stderr or "").strip()
                first2 = text2.splitlines()[0] if text2 else ""
                if res2.returncode == 0:
                    # We don't know the exact resolved path in this mode.
                    return True, first2, None
            except Exception:
                pass

        return False, first_line, resolved

    # Some tools print version to stderr.
    return True, first_line, resolved

# Check CUDA availability in PyTorch
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        print(f"  ✓ CUDA available: {device_name}")
        print(f"    Compute Capability: ({compute_cap[0]}, {compute_cap[1]})")
    else:
        print(f"  ✗ CUDA not available (CPU-only mode)")
except Exception as e:
    print(f"  ⚠ CUDA check failed: {e}")

# Check Numba CUDA availability
try:
    from numba import cuda
    numba_cuda_available = cuda.is_available()
    if numba_cuda_available:
        cc = cuda.get_current_device().compute_capability
        print(f"  ✓ Numba CUDA available: compute capability {cc[0]}.{cc[1]}")
    else:
        print(f"  ✗ Numba CUDA not available")
except Exception as e:
    print(f"  ⚠ Numba CUDA check failed: {e}")

# Check ncu (Nsight Compute)
try:
    ok, version_line, resolved = _run_tool_version("ncu", timeout_s=5)
    if ok:
        suffix = f" (at {resolved})" if resolved else ""
        print(f"  ✓ Nsight Compute (ncu): {version_line}{suffix}")

        # CUPTI counter permissions smoke test (Windows often requires Admin).
        try:
            from profiling.ncu_utils import ncu_metric_smoke_test

            smoke = ncu_metric_smoke_test(timeout_s=30)
            if smoke.ok:
                print("  ✓ CUPTI perf counters via ncu: OK")
            else:
                if smoke.reason in {"permission_denied", "permission_denied_not_admin"}:
                    print("  ⚠ CUPTI perf counters via ncu: PERMISSION DENIED")
                    if os.name == "nt":
                        print("    Fix: run Command Prompt as Administrator, then re-run ncu.")
                else:
                    print(f"  ⚠ CUPTI perf counters via ncu: {smoke.reason}")
        except Exception as e:
            print(f"  ⚠ CUPTI perf counters via ncu: smoke test failed ({e})")
    else:
        # Print a short hint; PATH can differ between terminal and VS Code.
        print("  ✗ Nsight Compute (ncu) not found or not runnable from this Python process")
        if resolved:
            print(f"    Resolved to: {resolved} (but failed to run)")
        if os.name == "nt":
            print("    Hint: If `ncu --version` works in your terminal but fails here, your Python process PATH/PATHEXT may differ.")
except Exception as e:
    print(f"  ✗ Nsight Compute (ncu) not found: {e}")

# Check ptxas (NVIDIA assembler)
try:
    import subprocess
    result = subprocess.run(["ptxas", "--version"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        version_line = result.stdout.strip().split('\n')[0]
        print(f"  ✓ PTXAS (NVIDIA Assembler): {version_line}")
    else:
        print(f"  ✗ PTXAS not found in PATH")
except Exception as e:
    print(f"  ✗ PTXAS not found: {e}")

# Check CUDA runtime
try:
    import subprocess
    result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        version_line = result.stdout.strip().split('\n')[-1]
        print(f"  ✓ NVCC (CUDA Compiler): {version_line}")
    else:
        print(f"  ✗ NVCC (CUDA Compiler) not found")
except Exception as e:
    print(f"  ✗ NVCC not found: {e}")

# Check nvidia-smi
try:
    import subprocess
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", 
                            "--format=csv,noheader"], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"  ✓ NVIDIA SMI: {result.stdout.strip()}")
    else:
        print(f"  ✗ nvidia-smi query failed")
except Exception as e:
    print(f"  ✗ nvidia-smi not found: {e}")

print()
print("=" * 70)
print(f"Summary: {len(results['installed'])} installed, "
      f"{len(results['missing'])} missing, "
      f"{len(results['errors'])} errors")
print("=" * 70)

if results["missing"]:
    print("\n[MISSING PACKAGES]")
    for pkg in results["missing"]:
        print(f"  - {pkg}")
    print("\nInstall with: pip install <package_name>")

if results["errors"]:
    print("\n[PACKAGES WITH ERRORS]")
    for pkg, error in results["errors"]:
        print(f"  - {pkg}: {error}")

# Exit code: 0 if all OK, 1 if missing/errors
exit_code = 1 if (results["missing"] or results["errors"]) else 0
sys.exit(exit_code)
