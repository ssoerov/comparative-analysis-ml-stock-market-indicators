"""
Строит ГОСТ‑графики по мастер‑таблицам: факт целевой переменной и тестовые прогнозы
моделей (SARIMAX, CatBoost, LSTM_att, Hybrid) для каждого фолда и тикера.

Использует:
- master_table_{TICKER}.csv для фактического ряда (y_dclose / y_logret);
- preds файлы (outputs[/_logret]/preds/{TICKER}_f{fold}.csv) для тестовых прогнозов
  и границ окон форкаста.

Результаты сохраняются в outputs/master_plots/{ticker}_{target}_f{fold}.png
и дублируются в outputs_logret/master_plots для logret.
"""

import argparse
import os
import sys
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless / sandbox
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# локальный импорт
sys.path.append(os.getcwd())
from vkr_fast.plotting import apply_gost_style  # noqa: E402

MODELS = ["SARIMAX", "CatBoost", "LSTM_att", "Hybrid"]
COLORS = {
    "SARIMAX": "#1f77b4",
    "CatBoost": "#d62728",
    "LSTM_att": "#2ca02c",
    "Hybrid": "#9467bd",
}


def _load_preds(pred_path: str) -> pd.DataFrame:
    df = pd.read_csv(pred_path, parse_dates=["Datetime"])
    return df


def _forecast_window(df_pred: pd.DataFrame) -> pd.Interval:
    """Границы тестового окна по preds (минимальная/максимальная дата)."""
    return pd.Interval(df_pred["Datetime"].min(), df_pred["Datetime"].max(), closed="both")


def plot_fold(
    master: pd.DataFrame,
    preds: pd.DataFrame,
    tk: str,
    target: str,
    fold: str,
    out_dir: str,
) -> str:
    apply_gost_style()
    fig, ax = plt.subplots(figsize=(11, 4))

    # факт
    ax.plot(master["Datetime"], master[f"y_{target}"], color="#444444", linewidth=1.4, label="Факт")

    # окно прогноза
    win = _forecast_window(preds)
    ax.axvspan(win.left, win.right, color="#f0f0f0", alpha=0.8, label="Окно прогноза")

    # модели — только на тестовом окне
    for mdl in MODELS:
        if mdl not in preds.columns:
            continue
        part = preds[["Datetime", mdl]].dropna()
        if part.empty:
            continue
        ax.plot(part["Datetime"], part[mdl], linewidth=1.6, label=mdl, color=COLORS.get(mdl))

    # оформление
    ax.set_xlabel("Дата и время (UTC)")
    ax.set_ylabel("Значение")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0.0, frameon=True)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{tk}_{target}_f{fold}.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def build_plots(master_path: str, preds_dir: str, out_dir: str, target: str) -> List[str]:
    master = pd.read_csv(master_path, parse_dates=["Datetime"])
    tk = os.path.basename(master_path).split("_")[-1].split(".")[0]
    saved: List[str] = []
    # какой набор фолдов есть в preds_dir
    pred_files = sorted([p for p in os.listdir(preds_dir) if p.startswith(f"{tk}_f") and p.endswith(".csv")])
    for pf in pred_files:
        fold = pf.split("_f")[-1].split(".")[0]
        preds = _load_preds(os.path.join(preds_dir, pf))
        saved.append(plot_fold(master, preds, tk, target, fold, out_dir))
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", required=True, help="Путь к master_table_{TICKER}.csv")
    ap.add_argument("--preds-dir", required=True, help="Каталог с preds/{ticker}_f*.csv")
    ap.add_argument("--out-dir", default="outputs/master_plots", help="Куда сохранять графики")
    ap.add_argument("--target", choices=["dclose", "logret"], required=True, help="Целевая переменная")
    args = ap.parse_args()

    saved = build_plots(args.master, args.preds_dir, args.out_dir, args.target)
    print(f"Saved {len(saved)} plots to {args.out_dir}")


if __name__ == "__main__":
    main()
