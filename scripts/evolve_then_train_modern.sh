#!/usr/bin/env bash
# Evolve modern MoE+attention genomes, then train the winner (final pipeline).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$(pwd)/.venv/bin/python" ]]; then
    PYTHON="$(pwd)/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python)"
  fi
fi

"$PYTHON" -m evo_lm evolve --config configs/evolve_modern_moe_attn_v1.yaml
"$PYTHON" -m evo_lm train --config configs/train_modern_moe_attn_v1.yaml
echo "hybrid best: runs/train_modern_moe_attn_v1/best.pt"
