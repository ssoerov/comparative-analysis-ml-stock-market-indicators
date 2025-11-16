#!/usr/bin/env bash
set -euo pipefail

# Detect conda and prefer running inside vkr311 env
if command -v conda >/dev/null 2>&1; then
  PY=(conda run -n vkr311 python)
else
  echo "[warn] conda not found; falling back to system python"
  PY=(python)
fi

# Full run: 5 folds, LSTM/Hybrid, with GARCH feature (optional)
"${PY[@]}" cli.py \
  --tickers IMOEX \
  --outer-folds 5 \
  --epochs 8 \
  --use-garch \
  --garch-mode feature \
  --validate
