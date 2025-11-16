import os
from typing import Dict, List

import numpy as np
import pandas as pd

# local imports
import sys
sys.path.append(os.getcwd())
from vkr_fast.config import Paths, TimeParams, TICKERS
from vkr_fast.data import fetch_moex, yahoo_csv, keyrate_series
from vkr_fast.plotting import apply_gost_style
from vkr_fast.mpl_config import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.api import OLS, add_constant

try:
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import ta
    TA_OK = True
except Exception:
    TA_OK = False


def _load_raw(paths: Paths, timep: TimeParams, tickers: Dict[str, tuple]) -> Dict[str, pd.DataFrame]:
    brent = yahoo_csv(paths.data_dir, "Brent_yahoo_1h_2023-01-01_2025-04-01 (1).csv")
    usdr = yahoo_csv(paths.data_dir, "USD_RUB_yahoo_1h_2023-01-01_2025-04-01.csv")
    krate = keyrate_series()
    period = slice(timep.period_start, timep.period_end - pd.Timedelta(seconds=1))

    out = {}
    for tk in tickers:
        moex = fetch_moex(
            paths.cache_dir, tk, *tickers[tk], timep.start_raw, timep.end_raw, timep.interval_minutes, use_cache_only=False
        )
        idx = moex.index
        ext = pd.DataFrame(index=idx)
        ext["Brent"] = brent["Close"].reindex(idx, method="ffill")
        ext["USD"] = usdr["Close"].reindex(idx, method="ffill")
        ext["KeyRate"] = krate["KeyRate"].reindex(idx, method="ffill")
        ext.bfill(limit=1, inplace=True)
        out[tk] = moex.join(ext, how="left").loc[period]
    return out


def _mk_out_dir(base: str = "outputs/eda") -> str:
    os.makedirs(base, exist_ok=True)
    return base


def _desc_stats(s: pd.Series) -> Dict[str, float]:
    s = pd.Series(s).astype(float)
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
        "min": float(s.min()),
        "p25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "max": float(s.max()),
        "skew": float(s.skew()) if len(s) > 2 else float("nan"),
        "kurt": float(s.kurt()) if len(s) > 3 else float("nan"),
    }


def _hac_slope_test(y: pd.Series) -> Dict[str, float]:
    # тренд уровня: OLS на времени с HAC (Newey-West) ковариацией
    if len(y) < 10:
        return {"beta": float("nan"), "pval": float("nan")}
    t = np.arange(len(y), dtype=float)
    X = add_constant(t)
    try:
        res = OLS(y.values.astype(float), X).fit(cov_type="HAC", cov_kwds={"maxlags": min(10, len(y)//10 + 1)})
        beta = float(res.params[1])
        pval = float(res.pvalues[1])
    except Exception:
        beta, pval = float("nan"), float("nan")
    return {"beta": beta, "pval": pval}


def _variance_trend_test(y: pd.Series) -> Dict[str, float]:
    # тенденция дисперсии: регрессия квадратов на времени с HAC
    r2 = (y.values.astype(float)) ** 2
    if len(r2) < 10:
        return {"beta_var": float("nan"), "pval_var": float("nan")}
    t = np.arange(len(r2), dtype=float)
    X = add_constant(t)
    try:
        res = OLS(r2, X).fit(cov_type="HAC", cov_kwds={"maxlags": min(10, len(r2)//10 + 1)})
        beta = float(res.params[1])
        pval = float(res.pvalues[1])
    except Exception:
        beta, pval = float("nan"), float("nan")
    return {"beta_var": beta, "pval_var": pval}


def main():
    apply_gost_style()
    paths = Paths()
    timep = TimeParams()
    tickers = TICKERS
    eda_dir = _mk_out_dir()

    raw = _load_raw(paths, timep, tickers)

    # агрегированная сводка по тикерам
    rows: List[Dict[str, float]] = []

    # Корреляции по dClose между тикерами
    dclose_df = pd.DataFrame({tk: df["Close"].diff().dropna() for tk, df in raw.items()}).dropna()
    corr = dclose_df.corr()
    corr.to_csv(os.path.join(eda_dir, "corr_dClose.csv"))

    # Кластеризация по корреляциям
    if SCIPY_OK and len(corr) >= 2:
        dist = 1 - corr
        # сделать симметричную «distance» матрицу в condensed form
        dcond = squareform(dist.values, checks=False)
        Z = linkage(dcond, method="average")
        plt.figure(figsize=(8, 4))
        dendrogram(Z, labels=corr.index.tolist(), leaf_rotation=45)
        plt.title("Дендрограмма кластеризации (dClose)")
        plt.tight_layout()
        plt.savefig(os.path.join(eda_dir, "cluster_dendrogram.png"))
        plt.close()

    for tk, df in raw.items():
        if "Close" not in df:
            continue
        close = df["Close"].astype(float)
        dclose = close.diff().dropna()

        # Описательная статистика
        desc_c = _desc_stats(close)
        desc_dc = _desc_stats(dclose)

        # Тесты стационарности/зависимости
        tests = {}
        try:
            adf_stat, adf_p, _, _, _, _ = (*adfuller(dclose.values, autolag="AIC"),)
            tests.update({"adf_stat": float(adf_stat), "adf_p": float(adf_p)})
        except Exception:
            tests.update({"adf_stat": np.nan, "adf_p": np.nan})
        try:
            kpss_stat, kpss_p, _, _ = kpss(dclose.values, regression="c", nlags="auto")
            tests.update({"kpss_stat": float(kpss_stat), "kpss_p": float(kpss_p)})
        except Exception:
            tests.update({"kpss_stat": np.nan, "kpss_p": np.nan})
        try:
            lb = acorr_ljungbox(dclose.values, lags=[10], return_df=True)
            tests.update({"lb_stat": float(lb["lb_stat"].iloc[0]), "lb_p": float(lb["lb_pvalue"].iloc[0])})
        except Exception:
            tests.update({"lb_stat": np.nan, "lb_p": np.nan})
        try:
            arch = het_arch(dclose.values, nlags=5)
            tests.update({"arch_lm": float(arch[0]), "arch_p": float(arch[1])})
        except Exception:
            tests.update({"arch_lm": np.nan, "arch_p": np.nan})

        trend = _hac_slope_test(close)
        vartrend = _variance_trend_test(dclose)

        rows.append({
            "Tk": tk,
            **{f"c_{k}": v for k, v in desc_c.items()},
            **{f"dc_{k}": v for k, v in desc_dc.items()},
            **tests,
            **trend,
            **vartrend,
        })

        # Отдельные фигуры по ГОСТ: уровень, приращения, скользящая σ, АКФ
        dt = close.index
        p_start, p_end = dt.min(), dt.max()

        # 1) Уровень (цена закрытия)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dt, close, label="Цена закрытия")
        ax.set_title(f"{tk}: Динамика ряда цены закрытия за период {p_start:%Y-%m-%d} — {p_end:%Y-%m-%d}")
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("Цена закрытия")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, f"{tk}_eda_close.png"))
        plt.close(fig)

        # 2) Приращения (ΔЦена)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(dclose.index, dclose, label="ΔЦена")
        ax.set_title(f"{tk}: Динамика приращений цены (ΔЦена)")
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("Изменение цены (ΔЦена)")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, f"{tk}_eda_dclose.png"))
        plt.close(fig)

        # 3) Скользящая σ
        roll_win = 50
        roll_sig = dclose.rolling(roll_win).std()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(roll_sig.index, roll_sig, label=f"Скользящая σ (окно {roll_win})")
        ax.set_title(f"{tk}: Скользящая оценка волатильности σ (окно {roll_win})")
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("σ")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, f"{tk}_eda_rolling_sigma.png"))
        plt.close(fig)

        # 4) АКФ Δцены
        from statsmodels.graphics.tsaplots import plot_acf
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(dclose.values, lags=40, ax=ax)
        ax.set_title(f"{tk}: АКФ Δцены")
        ax.set_xlabel("Лаг")
        ax.set_ylabel("Корреляция")
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(eda_dir, f"{tk}_eda_acf_dclose.png"))
        plt.close(fig)

        # 5) Коррелограмма (heatmap) факторов и показателей
        # Формируем матрицу показателей с русскими именами
        data_cols = {}
        data_cols["Цена закрытия"] = close
        data_cols["ΔЦена"] = close.diff()
        if "Brent" in df:
            data_cols["Brent"] = df["Brent"].astype(float)
        if "USD" in df:
            data_cols["USD"] = df["USD"].astype(float)
        if "KeyRate" in df:
            data_cols["Ключевая ставка"] = df["KeyRate"].astype(float)
        if TA_OK:
            try:
                data_cols["RSI(50)"] = ta.momentum.rsi(close, window=50)
            except Exception:
                pass
            try:
                data_cols["ATR(50)"] = ta.volatility.average_true_range(df.get("High"), df.get("Low"), df.get("Close"), window=50)
            except Exception:
                pass
        # Скользящая σ по Δцене (50)
        data_cols["σ (окно 50)"] = close.diff().rolling(50).std()

        corr_df = pd.concat(data_cols, axis=1)
        corr_df = corr_df.replace([np.inf, -np.inf], np.nan).dropna()
        if corr_df.shape[1] >= 2 and len(corr_df) > 5:
            C = corr_df.corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(C.shape[1]))
            ax.set_xticklabels(C.columns, rotation=45, ha="right")
            ax.set_yticks(range(C.shape[1]))
            ax.set_yticklabels(C.columns)
            for i in range(C.shape[0]):
                for j in range(C.shape[1]):
                    ax.text(j, i, f"{C.values[i, j]:.2f}", va="center", ha="center", fontsize=9, color="black")
            p_start, p_end = close.index.min(), close.index.max()
            ax.set_title(f"{tk}: Коррелограмма показателей и факторов (Пирсон) за период {p_start:%Y-%m-%d} — {p_end:%Y-%m-%d}")
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Коэффициент корреляции Пирсона")
            ax.grid(False)
            fig.tight_layout()
            fig.savefig(os.path.join(eda_dir, f"{tk}_eda_corr_heatmap.png"))
            plt.close(fig)

    pd.DataFrame(rows).to_csv(os.path.join(eda_dir, "summary.csv"), index=False)
    print("EDA готово: outputs/eda/")


if __name__ == "__main__":
    main()
