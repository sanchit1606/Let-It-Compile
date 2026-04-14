"""Quick diagnostic script to check CUDA/GPU availability."""

import torch
import numpy as np

print("=" * 80)
print("CUDA/GPU Diagnostic Check")
print("=" * 80)

print(f"\n1. PyTorch version: {torch.__version__}")
print(f"2. CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"3. CUDA version (PyTorch sees): {torch.version.cuda}")
    print(f"4. Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n   GPU {i}:")
        print(f"     - Name: {torch.cuda.get_device_name(i)}")
        print(f"     - Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"     - Compute Capability: {torch.cuda.get_device_capability(i)}")
    
    # Test a simple CUDA operation
    print(f"\n5. Testing CUDA operation...")
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"   ✓ CUDA operations work!")
        print(f"   ✓ Result shape: {z.shape}, device: {z.device}")
    except Exception as e:
        print(f"   ✗ CUDA operation failed: {e}")
else:
    print("\n⚠️  CUDA is NOT available!")
    print("   PyTorch will use CPU (training will be VERY slow)")
    print("\n   To fix this:")
    print("   1. Ensure NVIDIA GPU drivers are installed")
    print("   2. Reinstall PyTorch with CUDA support:")
    print("      conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia")

print("\n" + "=" * 80)
