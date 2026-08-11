#!/usr/bin/env bash
# Train the ~7M Transformer baseline under the same modern dialogue protocol.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -x "$(pwd)/.venv/bin/python" ]]; then
    PYTHON="$(pwd)/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python)"
  fi
fi

"$PYTHON" -m baselines.transformer.train --config baselines/transformer/configs/train_7m_modern.yaml
echo "transformer best: runs/train_transformer_baseline_v1/best.pt"
