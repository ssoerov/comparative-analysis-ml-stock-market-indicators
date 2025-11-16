import os
import math
from typing import Dict, List

import numpy as np
import pandas as pd
from .mpl_config import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import shapiro, probplot

from .plotting import apply_gost_style


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _residual_tests(err: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    err = np.asarray(err)
    if len(err) >= 8:
        try:
            stat, p = shapiro(err)
            out["shapiro_stat"], out["shapiro_p"] = float(stat), float(p)
        except Exception:
            pass
    # Ljung–Box for lags 10
    try:
        lb = acorr_ljungbox(err, lags=[10], return_df=True)
        out["ljungbox_stat"], out["ljungbox_p"] = float(lb["lb_stat"].iloc[0]), float(lb["lb_pvalue"].iloc[0])
    except Exception:
        pass
    # ARCH LM
    try:
        arch = het_arch(err, nlags=5)
        out["arch_lm_stat"], out["arch_lm_p"] = float(arch[0]), float(arch[1])
    except Exception:
        pass
    # Durbin–Watson
    try:
        out["durbin_watson"] = float(durbin_watson(err))
    except Exception:
        pass
    return out


def _equity_from_preds(close: np.ndarray, yhat: np.ndarray, dt: np.ndarray, fee: float, threshold: float = 0.0, slippage: float = 0.0) -> Dict[str, np.ndarray]:
    # close input must align with yhat such that returns are computed between aligned timestamps
    log_ret = np.log(close[1:] / close[:-1])
    pos = np.where(yhat > threshold, 1, np.where(yhat < -threshold, -1, 0))
    pos = pos[: len(log_ret)]  # guard
    dpos = np.diff(np.insert(pos, 0, 0))
    cost = (fee + slippage) * np.abs(dpos)
    strat = pos * log_ret - cost
    eq = np.exp(np.insert(strat.cumsum(), 0, 0))
    dd = np.maximum.accumulate(eq) - eq
    return {"equity": eq, "drawdown": dd}


def generate_reports(
    preds_df: pd.DataFrame,
    out_dir: str,
    tk: str,
    fold: int,
    fee: float,
    threshold: float = 0.0,
    slippage: float = 0.0,
    gost: bool = True,
    history=None,
) -> Dict[str, List]:
    """Generate diagnostics and figures for a given predictions dataframe.

    preds_df columns expected: Datetime, y_true, Close, and any model columns.
    """
    if gost:
        apply_gost_style()
    rep_dir = os.path.join(out_dir, "reports")
    _ensure_dir(rep_dir)

    # Determine model columns
    skip_cols = {"Datetime", "y_true", "Close", "Close_prev", "Sigma"}
    cols = [c for c in preds_df.columns if c not in skip_cols]
    results = {"models": []}

    # Full overlay: факт vs все модели на одном графике (с отсечкой начала прогноза)
    if history is not None:
        hist_dt, hist_y, fc_start_dt = history
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(hist_dt, hist_y, label="Факт (ΔЦена)", color="black", linewidth=1.8)
        fc_dt = pd.to_datetime(preds_df["Datetime"])
        for c in cols:
            ax.plot(fc_dt, preds_df[c], label=f"Прогноз {c}", linewidth=1.8, alpha=0.9, solid_capstyle="round")
        ax.axvline(fc_start_dt, color="#666666", linestyle="--", linewidth=1.2, label="Старт прогноза")
        ax.set_title(f"{tk} — факт и прогнозы всех моделей, фолд {fold}")
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("Изменение цены (ΔЦена)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(rep_dir, f"{tk}_f{fold}_all_models.png"))
        plt.close(fig)

    # Line plots: actual vs predicted (по моделям)
    for c in cols:
        fig, ax = plt.subplots()
        dt = pd.to_datetime(preds_df["Datetime"]) 
        p_start, p_end = dt.min(), dt.max()
        ax.plot(dt, preds_df["y_true"], label="Факт (ΔЦена)")
        ax.plot(dt, preds_df[c], label=f"Прогноз ({c})")
        ax.set_title(f"{tk} — Динамика приращений (факт и прогноз {c}) за период {p_start:%Y-%m-%d} — {p_end:%Y-%m-%d}, фолд {fold}")
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("Изменение цены (ΔЦена)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(rep_dir, f"{tk}_f{fold}_actual_vs_{c}.png"))
        plt.close(fig)

    # Residual diagnostics and plots
    diags_rows = []
    # Series-level stationarity on target once per fold
    try:
        y = preds_df["y_true"].values
        adf_stat, adf_p, _, _, _, _ = (*adfuller(y, autolag="AIC"),)
        kpss_stat, kpss_p, _, _ = kpss(y, regression="c", nlags="auto")
        series_diag = pd.DataFrame(
            [{"Tk": tk, "Fold": fold, "Series": "y_true", "ADF_stat": adf_stat, "ADF_p": adf_p, "KPSS_stat": kpss_stat, "KPSS_p": kpss_p}]
        )
        series_diag.to_csv(os.path.join(rep_dir, f"{tk}_f{fold}_series_stationarity.csv"), index=False)
    except Exception:
        pass
    for c in cols:
        err = preds_df["y_true"].values - preds_df[c].values
        tests = _residual_tests(err)
        # Residual stationarity
        try:
            adf_stat, adf_p, _, _, _, _ = (*adfuller(err, autolag="AIC"),)
            tests["resid_adf_stat"], tests["resid_adf_p"] = float(adf_stat), float(adf_p)
        except Exception:
            pass
        try:
            kpss_stat, kpss_p, _, _ = kpss(err, regression="c", nlags="auto")
            tests["resid_kpss_stat"], tests["resid_kpss_p"] = float(kpss_stat), float(kpss_p)
        except Exception:
            pass
        diags_rows.append({"Tk": tk, "Fold": fold, "Model": c, **tests})

        # ACF
        fig, ax = plt.subplots()
        plot_acf(err, lags=40, ax=ax)
        ax.set_title(f"{tk} — АКФ остатков ({c}), фолд {fold}")
        ax.set_xlabel("Лаг")
        ax.set_ylabel("Корреляция")
        fig.tight_layout()
        fig.savefig(os.path.join(rep_dir, f"{tk}_f{fold}_acf_{c}.png"))
        plt.close(fig)

        # Histogram
        fig, ax = plt.subplots()
        ax.hist(err, bins=30, color="#4C72B0", edgecolor="black", alpha=0.8)
        ax.set_title(f"{tk} — Распределение остатков ({c}), фолд {fold}")
        ax.set_xlabel("Ошибка (факт − прогноз)")
        ax.set_ylabel("Частота")
        fig.tight_layout()
        fig.savefig(os.path.join(rep_dir, f"{tk}_f{fold}_hist_{c}.png"))
        plt.close(fig)

        fig, ax = plt.subplots()
        (osm, osr), (slope, intercept, _) = probplot(err, dist="norm")
        ax.scatter(osm, osr, s=12, color="#1f77b4")
        min_x, max_x = np.min(osm), np.max(osm)
        ax.plot([min_x, max_x], [intercept + slope * min_x, intercept + slope * max_x], color="#d62728", linewidth=1.2)
        ax.set_title(f"{tk} — Квантильный график остатков ({c}), фолд {fold}")
        ax.set_xlabel("Теоретические квантили N(0,1)")
        ax.set_ylabel("Наблюдаемые квантили")
        fig.tight_layout()
        fig.savefig(os.path.join(rep_dir, f"{tk}_f{fold}_qq_{c}.png"))
        plt.close(fig)

    diags = pd.DataFrame(diags_rows)
    diags.to_csv(os.path.join(rep_dir, f"{tk}_f{fold}_diagnostics.csv"), index=False)

    # Equity curves for all models in one figure
    fig, ax = plt.subplots()
    for c in cols:
        # to compute equity, we need close aligned with predictions; append the last close for same length+1
        close_curr = preds_df["Close"].values
        if "Close_prev" in preds_df.columns and len(preds_df["Close_prev"].values) == len(close_curr):
            close_seq = np.concatenate(([preds_df["Close_prev"].values[0]], close_curr))
        else:
            close_seq = np.concatenate(([close_curr[0]], close_curr))
        dt = pd.to_datetime(preds_df["Datetime"]).values
        eq = _equity_from_preds(close_seq, preds_df[c].values, dt, fee, threshold=threshold, slippage=slippage)
        ax.plot(eq["equity"], label=f"{c}")
    ax.set_title(f"{tk} — Динамика капитала по моделям, фолд {fold}")
    ax.set_xlabel("Номер шага")
    ax.set_ylabel("Капитал (начальный = 1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(rep_dir, f"{tk}_f{fold}_equity.png"))
    plt.close(fig)

    # Plot sigma (if present)
    if "Sigma" in preds_df.columns:
        fig, ax = plt.subplots()
        dt = pd.to_datetime(preds_df["Datetime"]) 
        ax.plot(dt, preds_df["Sigma"], label="σ (GARCH)")
        ax.set_title(f"{tk} — Оценка волатильности σ (GARCH), фолд {fold}")
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("σ")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(rep_dir, f"{tk}_f{fold}_sigma.png"))
        plt.close(fig)

    results["models"] = cols
    return results
