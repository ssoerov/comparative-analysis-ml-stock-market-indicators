#!/usr/bin/env bash
set -euo pipefail

# Единый запуск: обучение + EDA + консолидация + отчёт

# Выбор Python: предпочтительно через conda-окружение vkr311
if command -v conda >/dev/null 2>&1; then
  PY=(conda run -n vkr311 python)
else
  echo "[warn] conda не найден — используется системный python"
  PY=(python)
fi

echo "[1/4] Модели и прогнозы (walk-forward)"
"${PY[@]}" cli.py \
  --tickers IMOEX \
  --outer-folds 5 \
  --epochs 8 \
  --use-garch \
  --garch-mode feature \
  --validate

echo "[2/4] EDA по рядам"
"${PY[@]}" tools/run_eda.py

echo "[3/4] Кластеризация режимов"
"${PY[@]}" tools/cluster_regimes.py --k 5 --win-vol 50

echo "[4/6] VAR/VARMAX (многомерное моделирование)"
"${PY[@]}" tools/run_var.py --use-varmax --exog-lags 1

echo "[5/6] Консолидация метрик и экономики"
"${PY[@]}" tools/consolidate_from_preds.py --fee 0.002 --slippage 0.0002 --threshold 0.0

echo "[6/6] Сборка отчёта"
"${PY[@]}" tools/build_report.py

echo "Готово. Отчёт: REPORT.md. Артефакты: ./outputs, ./outputs/eda, ./outputs/clustering, ./outputs/consolidated"
