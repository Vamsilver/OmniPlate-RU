import os
import sys
import torch

print("=" * 50)
print("  GPU Diagnostic Tool (Volga IT 2026)")
print("=" * 50)
print("PyTorch Version: ", torch.__version__)
print("CUDA Available:  ", torch.cuda.is_available())

if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    archs = torch.cuda.get_arch_list()
    print(f"GPU Device:       {name}")
    print(f"Capability:       {cap} (sm_{cap[0]}{cap[1]})")
    print(f"Supported Archs:  {archs}")
    print("\nTesting simple tensor operation on GPU...")
    try:
        x = torch.randn(100, 100, device="cuda")
        y = x @ x
        print("[SUCCESS] PyTorch successfully executed on CUDA GPU!")
    except Exception as e:
        print(f"[FAIL] GPU Execution Error: {e}")
        print("\nChecking if CUDA_FORCE_PTX_JIT=1 helps...")
