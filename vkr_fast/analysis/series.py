import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from ..mpl_config import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from ..plotting import apply_gost_style


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def chain_growth_report(close: pd.Series, tk: str, out_dir: str, freq: str = "D") -> Dict[str, str]:
    """Build chain increment/growth diagnostics for a closing price series."""
    _ensure_dir(out_dir)
    apply_gost_style()

    series = close.astype(float).resample(freq).last().dropna()
    df = pd.DataFrame({"Close": series})
    prev = df["Close"].shift(1)
    prev = prev.where(prev != 0.0, np.nan)
    df["ChainIncrement"] = df["Close"] - prev
    df["ChainGrowthCoef"] = df["Close"] / prev
    df["RelativeGrowth"] = (df["Close"] - prev) / prev
    df["RelativeGrowthPct"] = df["RelativeGrowth"] * 100
    df["ChainIndex"] = (1 + df["RelativeGrowth"].fillna(0)).cumprod()
    base_level = float(df["Close"].iloc[0])
    df["BaseAbsolute"] = df["Close"] - base_level
    df["BaseRelativeCoef"] = df["Close"] / base_level
    df["BaseRelativePct"] = (df["BaseRelativeCoef"] - 1.0) * 100.0
    df["ChainAbsoluteIndex"] = df["ChainIncrement"]
    df["ChainRelativeIndex"] = df["ChainGrowthCoef"]
    df["StructuralShiftAbsolute"] = df["BaseAbsolute"].diff()
    df["StructuralShiftRelativePct"] = (df["StructuralShiftAbsolute"] / prev) * 100.0
    df["StructuralShiftRelativePct"] = df["StructuralShiftRelativePct"].fillna(0.0)
    df = df.dropna()

    chain_path = os.path.join(out_dir, f"{tk}_chain_growth_{freq}.csv")
    df.to_csv(chain_path, index_label="Date")

    monthly = df.resample("ME").agg(
        Close=("Close", "last"),
        ChainIncrement=("ChainIncrement", "sum"),
        MeanRelGrowthPct=("RelativeGrowthPct", "mean"),
        Volatility=("ChainIncrement", "std"),
    )
    idx = monthly.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)
    monthly.index = idx.to_period("M").astype(str)
    summary_path = os.path.join(out_dir, f"{tk}_chain_growth_summary.csv")
    monthly.reset_index().rename(columns={"index": "Month"}).to_csv(summary_path, index=False)

    # aggregated stats for report text
    agg = {
        "mean_abs_increment": float(df["ChainIncrement"].abs().mean()),
        "median_growth_pct": float(df["RelativeGrowthPct"].median()),
        "max_growth_pct": float(df["RelativeGrowthPct"].max()),
        "min_growth_pct": float(df["RelativeGrowthPct"].min()),
        "mean_structural_shift_pct": float(df["StructuralShiftRelativePct"].abs().mean()),
    }
    agg_path = os.path.join(out_dir, f"{tk}_chain_growth_stats.json")
    with open(agg_path, "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2)

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    idx = df.index
    axes[0].plot(idx, df["Close"], label="Цена закрытия", color="#1f77b4")
    axes[0].set_title(f"{tk}: Цепной анализ (Цена закрытия)")
    axes[0].set_ylabel("Close")
    axes[0].legend()

    axes[1].bar(idx, df["ChainIncrement"], color=np.where(df["ChainIncrement"] >= 0, "#2ca02c", "#d62728"), width=0.8)
    axes[1].set_title("Цепные приросты (Δ)")
    axes[1].set_ylabel("Δ")

    axes[2].plot(idx, df["RelativeGrowthPct"], label="Относительный рост, %", color="#ff7f0e")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Δ%")
    axes[2].set_title("Цепной рост (относительный, %)")
    axes[2].legend()

    axes[3].plot(idx, df["BaseRelativePct"], label="Базисный рост, %", color="#9467bd")
    axes[3].plot(idx, df["StructuralShiftRelativePct"], label="Индекс структурного сдвига, %", color="#8c564b")
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_ylabel("%")
    axes[3].set_title("Базисные индексы и структурный сдвиг")
    axes[3].legend()
    axes[3].set_xlabel("Дата")

    fig.tight_layout()
    plot_path = os.path.join(out_dir, f"{tk}_chain_growth.png")
    fig.savefig(plot_path)
    plt.close(fig)

    # STL decomposition (trend + seasonal)
    series_full = close.astype(float).dropna()
    period = max(7, min(90, max(7, len(series_full) // 8))) or 24
    stl = STL(series_full, period=period, robust=True).fit()
    trend = stl.trend
    seasonal = stl.seasonal
    trend_fig = os.path.join(out_dir, f"{tk}_trend_component.png")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(trend.index, trend.values, color="#1f77b4")
    ax.set_title(f"{tk}: Трендовая компонента (STL)")
    ax.set_ylabel("Тренд")
    ax.set_xlabel("Дата")
    fig.tight_layout()
    fig.savefig(trend_fig)
    plt.close(fig)

    seasonal_fig = os.path.join(out_dir, f"{tk}_seasonal_component.png")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(seasonal.index, seasonal.values, color="#d62728")
    ax.set_title(f"{tk}: Сезонная компонента (STL)")
    ax.set_ylabel("Сезонность")
    ax.set_xlabel("Дата")
    fig.tight_layout()
    fig.savefig(seasonal_fig)
    plt.close(fig)

    # Autocorrelation plots для цепных приростов
    acf_fig = os.path.join(out_dir, f"{tk}_acf_chain.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(df["ChainIncrement"], lags=40, ax=ax)
    ax.set_title(f"{tk}: АКФ цепных приращений")
    ax.set_xlabel("Лаг")
    ax.set_ylabel("Корреляция")
    fig.tight_layout()
    fig.savefig(acf_fig)
    plt.close(fig)

    pacf_fig = os.path.join(out_dir, f"{tk}_pacf_chain.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_pacf(df["ChainIncrement"], lags=40, ax=ax, method="ywm")
    ax.set_title(f"{tk}: ЧАКФ цепных приращений")
    ax.set_xlabel("Лаг")
    ax.set_ylabel("Частичная корреляция")
    fig.tight_layout()
    fig.savefig(pacf_fig)
    plt.close(fig)

    return {
        "table": chain_path,
        "summary": summary_path,
        "stats": agg_path,
        "figure": plot_path,
        "trend_fig": trend_fig,
        "season_fig": seasonal_fig,
        "acf_fig": acf_fig,
        "pacf_fig": pacf_fig,
    }
