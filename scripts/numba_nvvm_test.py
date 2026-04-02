import os
from pathlib import Path
import numpy as np

# Prefer conda's CUDA 12.1 NVVM (just installed via cuda-nvcc)
# Fallback to system CUDA 13 if conda NVVM not found
conda_env_path = Path(os.environ.get("CONDA_PREFIX", ""))
cuda_root = None

# Try conda NVVM first
if conda_env_path.exists():
    conda_nvvm = conda_env_path / "Library" / "bin" / "nvvm.dll"
    conda_libdevice = conda_env_path / "Library" / "nvvm" / "libdevice"
    if conda_nvvm.exists() and conda_libdevice.exists():
        cuda_root = str(conda_env_path / "Library")
        print("Using conda CUDA 12.1 NVVM")

# Fallback to system CUDA 13
if cuda_root is None:
    cuda_root = os.environ.get("CUDA_HOME") or r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    print("Using system CUDA NVVM")

os.environ["CUDA_HOME"] = cuda_root
os.environ["NUMBA_CUDA_NVVM"] = os.path.join(cuda_root, "bin", "nvvm.dll")
os.environ["NUMBA_CUDA_LIBDEVICE"] = os.path.join(cuda_root, "nvvm", "libdevice")

print("CUDA_HOME:", os.environ["CUDA_HOME"])
print("NUMBA_CUDA_NVVM:", os.environ["NUMBA_CUDA_NVVM"], "exists:", os.path.exists(os.environ["NUMBA_CUDA_NVVM"]))
print("NUMBA_CUDA_LIBDEVICE:", os.environ["NUMBA_CUDA_LIBDEVICE"], "exists:", os.path.exists(os.environ["NUMBA_CUDA_LIBDEVICE"]))

from numba import cuda

print("cuda.is_available():", cuda.is_available())

@cuda.jit
def add(a, b, c):
    i = cuda.grid(1)
    if i < a.size:
        c[i] = a[i] + b[i]

n = 256
a = cuda.to_device(np.ones(n, dtype=np.float32))
b = cuda.to_device(np.ones(n, dtype=np.float32))
c = cuda.device_array(n, dtype=np.float32)

add[1, 256](a, b, c)
cuda.default_stream().synchronize()
print("OK:", float(c.copy_to_host()[0]))
