#!/usr/bin/env bash
# Install ROCm-enabled PyTorch into the project venv (after system ROCm is ready).
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Python: $(python -V)"
echo "Checking rocminfo..."
if command -v rocminfo >/dev/null 2>&1; then
  rocminfo | grep -E 'Marketing Name|Name:[[:space:]]*gfx' | head -20 || true
else
  echo "WARN: rocminfo not found. Finish scripts/setup_rocm.sh + reboot first."
fi

echo "Installing torch (ROCm 6.4 wheels, cp314)..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install --index-url https://download.pytorch.org/whl/rocm6.4 'torch==2.9.1'

# gfx1103 sometimes needs a close-relative override on older libraries
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"

python - <<'PY'
import os, torch
print("torch", torch.__version__)
print("hip", getattr(torch.version, "hip", None))
print("HSA_OVERRIDE_GFX_VERSION", os.environ.get("HSA_OVERRIDE_GFX_VERSION"))
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch does not see a HIP device yet.")
print("device", torch.cuda.get_device_name(0))
x = torch.randn(2048, 2048, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("matmul ok", tuple(y.shape), "dtype", y.dtype)
PY

echo "OK. For training, prefer:"
echo "  export HSA_OVERRIDE_GFX_VERSION=11.0.0"
echo "  unset EVO_VULKAN_FWD_ONLY EVO_DEVICE"
echo "  # then point evo train device to cuda/hip once runtime supports it"
