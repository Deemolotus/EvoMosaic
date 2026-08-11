#!/usr/bin/env bash
# Setup ROCm for Ryzen 200 / Radeon 780M (gfx1103) on Ubuntu 26.04.
# Official path: inbox kernel driver + amdgpu-install --no-dkms.
# Run in a real terminal (needs your sudo password):
#   bash scripts/setup_rocm.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ "$(id -u)" -eq 0 ]]; then
  echo "Please run as your normal user (script will sudo when needed)."
  exit 1
fi

. /etc/os-release
KERNEL="$(uname -r)"
echo "== Host =="
echo "  OS:     $PRETTY_NAME ($VERSION_CODENAME)"
echo "  Kernel: $KERNEL"
echo "  User:   $USER"
echo "  CPU:    $(lscpu | awk -F: '/Model name/{print $2; exit}' | xargs)"

if [[ "${VERSION_ID:-}" != "26.04" ]]; then
  echo "WARN: this script targets Ubuntu 26.04; you have ${VERSION_ID:-unknown}."
fi

if [[ "$KERNEL" != 7.0.* ]]; then
  echo "WARN: Ryzen ROCm on 26.04 expects GA kernel 7.0.*; current=$KERNEL"
fi

echo
echo "== [1/4] Install amdgpu-install (ROCm 7.13 / 31.30 resolute) =="
TMP="$(mktemp -d)"
DEB_URL="https://repo.radeon.com/amdgpu-install/31.30/ubuntu/resolute/amdgpu-install_31.30.313000-1_all.deb"
CACHED="$ROOT/.cache/amdgpu-install_31.30.313000-1_all.deb"
DEB="$TMP/amdgpu-install_31.30.313000-1_all.deb"
if [[ -f "$CACHED" ]]; then
  cp -f "$CACHED" "$DEB"
  echo "Using cached installer: $CACHED"
else
  wget -O "$DEB" "$DEB_URL"
fi
sudo apt update
sudo apt install -y "$DEB"

echo
echo "== [2/4] Install ROCm userspace (inbox driver, --no-dkms, gfx110x) =="
# amdgpu-install auto-detect picks gfx1103, but packages are family-suffixed
# as gfx110x (no amdrocm-gfx1103 metapackage). Force the family.
GFX_VER="${ROCM_GFXVERSION:-gfx110x}"
echo "Using --gfxversion=$GFX_VER"
# Ryzen APUs on Ubuntu 26.04 must NOT install amdgpu-dkms.
sudo amdgpu-install -y --usecase=rocm --no-dkms --gfxversion="$GFX_VER"

# If dkms slipped in, remove it.
if dpkg -l | grep -q '^ii\s\+amdgpu-dkms'; then
  echo "Removing unintended amdgpu-dkms..."
  sudo apt autoremove -y amdgpu-dkms dkms || true
fi

echo
echo "== [3/4] GPU access groups =="
sudo usermod -aG render,video "$USER"
echo "Added $USER to render,video (re-login/reboot needed for new sessions)."

echo
echo "== [4/4] Quick verification =="
if command -v rocminfo >/dev/null 2>&1; then
  # May fail in this shell if groups not refreshed; try with sg.
  if sg render -c 'rocminfo' 2>/dev/null | tee "$TMP/rocminfo.txt" | grep -qi 'Marketing Name\|Name:.*gfx'; then
    grep -E 'Marketing Name|Name:|Device Type|gfx' "$TMP/rocminfo.txt" | head -40
  else
    echo "rocminfo installed; if agent list is empty, reboot then re-run: rocminfo"
    rocminfo 2>&1 | head -40 || true
  fi
else
  echo "WARN: rocminfo not on PATH yet; open a new shell or check /opt/rocm/bin"
fi

# Optional: bump TTM shared memory later with amd-ttm if needed.
echo
echo "== Optional next: ROCm PyTorch in project venv =="
cat <<'EOF'
After reboot (recommended), from the project:

  cd ~/Desktop/evo
  source .venv/bin/activate
  pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
  pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.1

  python - <<'PY'
  import torch
  print("torch", torch.__version__)
  print("hip", getattr(torch.version, "hip", None))
  print("cuda_available", torch.cuda.is_available())
  if torch.cuda.is_available():
      print("device", torch.cuda.get_device_name(0))
      x = torch.randn(1024, 1024, device="cuda")
      print("matmul ok", (x @ x).shape)
  PY

If gfx1103 is rejected by a library, try:
  export HSA_OVERRIDE_GFX_VERSION=11.0.0
  # or 11.0.2 depending on the stack
EOF

echo
echo "DONE system ROCm install. Please reboot, then run the PyTorch steps above."
echo "Or: bash scripts/setup_rocm_torch.sh"
