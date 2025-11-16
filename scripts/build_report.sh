#!/usr/bin/env bash
set -euo pipefail

if command -v conda >/dev/null 2>&1; then
  PY=(conda run -n vkr311 python)
else
  echo "[warn] conda not found; falling back to system python"
  PY=(python)
fi

# Rebuild EDA, clustering, consolidated metrics, and the Markdown report
"${PY[@]}" tools/run_eda.py
"${PY[@]}" tools/cluster_regimes.py --k 3 --win-vol 50
"${PY[@]}" tools/consolidate_from_preds.py --fee 0.002 --slippage 0.0002 --threshold 0.0
"${PY[@]}" tools/build_report.py

echo "Report rebuilt -> REPORT.md"
