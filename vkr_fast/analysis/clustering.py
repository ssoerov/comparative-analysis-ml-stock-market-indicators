import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from ..mpl_config import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from ..plotting import apply_gost_style

try:
    import ta

    TA_OK = True
except Exception:  # pragma: no cover
    TA_OK = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_features(df: pd.DataFrame, win_vol: int = 50) -> pd.DataFrame:
    out = df.copy()
    out["ret"] = np.log(out["Close"].astype(float)).diff()
    out["vol"] = out["ret"].rolling(win_vol).std()
    if TA_OK:
        out["rsi"] = ta.momentum.rsi(out["Close"].astype(float), window=50)
        out["atr"] = ta.volatility.average_true_range(out["High"], out["Low"], out["Close"], window=50)
    else:
        out["rsi"] = np.nan
        out["atr"] = np.nan
    return out


def _plot_clusters(tk: str, df: pd.DataFrame, lab: np.ndarray, out_dir: str, names: Dict[int, str]) -> Dict[str, str]:
    from matplotlib.collections import LineCollection
    from matplotlib.colors import BoundaryNorm
    import matplotlib.dates as mdates

    apply_gost_style()

    dt = pd.to_datetime(df.index)
    x = mdates.date2num(dt.to_pydatetime())
    y = df["Close"].astype(float).values
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    if len(points) < 2:
        return {}
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_labels = lab[:-1]

    K = int(np.max(lab)) + 1
    cmap = plt.get_cmap("tab10", K)
    norm = BoundaryNorm(np.arange(-0.5, K + 0.5, 1), K)

    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(seg_labels.astype(float))
    lc.set_linewidth(1.6)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    y_min, y_max = np.nanmin(y), np.nanmax(y)
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
        ax.set_ylim(y_min, y_max)
    p_start, p_end = dt.min(), dt.max()
    ax.set_title(f"{tk} — Структура режимов ряда (k={K}) за период {p_start:%Y-%m-%d} — {p_end:%Y-%m-%d}")
    ax.set_xlabel("Дата и время (UTC)")
    ax.set_ylabel("Цена закрытия")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    cbar = fig.colorbar(lc, ax=ax, ticks=np.arange(0, K))
    cbar.set_label("Кластер")
    fig.tight_layout()
    price_fig = os.path.join(out_dir, f"{tk}_clusters_price.png")
    fig.savefig(price_fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    sc = ax.scatter(df["ret"], df["vol"], c=lab, cmap="tab10", s=8, alpha=0.7)
    ax.set_title(f"{tk} — Связка доходность–σ по кластерам")
    ax.set_xlabel("Лог‑доходность")
    ax.set_ylabel("Скользящая σ")
    fig.tight_layout()
    scatter_fig = os.path.join(out_dir, f"{tk}_clusters_scatter.png")
    fig.savefig(scatter_fig)
    plt.close(fig)

    shares = []
    labels = []
    for c in sorted(np.unique(lab)):
        shares.append(float((lab == c).mean()))
        labels.append(names.get(c, f"Кластер {c}"))
    pie_fig = os.path.join(out_dir, f"{tk}_cluster_pie.png")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(shares, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"{tk} — Доли режимов (структурный сдвиг)")
    fig.tight_layout()
    fig.savefig(pie_fig)
    plt.close(fig)

    return {"price_fig": price_fig, "scatter_fig": scatter_fig, "pie_fig": pie_fig}


def _regime_name(ret_mean: float, vol_mean: float, ret_lo: float, ret_hi: float, vol_lo: float, vol_hi: float) -> str:
    if ret_mean >= ret_hi and vol_mean <= vol_lo:
        return "Стабильный рост"
    if ret_mean >= ret_hi and vol_mean > vol_lo:
        return "Волатильный рост"
    if ret_mean <= ret_lo and vol_mean <= vol_lo:
        return "Плавное снижение"
    if ret_mean <= ret_lo and vol_mean > vol_lo:
        return "Турбулентное снижение"
    if abs(ret_mean) < (ret_hi - ret_lo) / 4 and vol_mean <= vol_hi:
        return "Боковая консолидация"
    return "Неустойчивый боковик"


def cluster_regimes_for_ticker(
    tk: str,
    df: pd.DataFrame,
    out_dir: str,
    k: int = 5,
    win_vol: int = 50,
    min_points: int = 100,
) -> Optional[Dict[str, str]]:
    """Cluster IMOEX regimes and persist ГОСТ compliant visuals."""
    _ensure_dir(out_dir)
    feat = _make_features(df, win_vol=win_vol)
    X = feat[["ret", "vol", "rsi", "atr"]].astype(float)
    X = X.replace([np.inf, -np.inf], np.nan).dropna()
    feat = feat.loc[X.index]
    if len(X) < max(min_points, k * 5):
        return None
    xsc = StandardScaler().fit(X)
    Z = xsc.transform(X)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(Z)
    sil = float(silhouette_score(Z, lab)) if k > 1 else float("nan")

    feat = feat.copy()
    feat["cluster"] = lab
    ret_vals = feat["ret"].dropna()
    vol_vals = feat["vol"].dropna()
    ret_lo, ret_hi = np.nanquantile(ret_vals, [0.33, 0.66])
    vol_lo, vol_hi = np.nanquantile(vol_vals, [0.33, 0.66])
    stats = []
    name_map: Dict[int, str] = {}
    for c in sorted(np.unique(lab)):
        mask = lab == c
        mean_ret = float(feat.loc[mask, "ret"].mean())
        mean_vol = float(feat.loc[mask, "vol"].mean())
        label = _regime_name(mean_ret, mean_vol, ret_lo, ret_hi, vol_lo, vol_hi)
        name_map[int(c)] = label
        stats.append(
            {
                "Tk": tk,
                "Cluster": int(c),
                "Share": float(mask.mean()),
                "Mean_ret": mean_ret,
                "Vol": mean_vol,
                "RSI": float(feat.loc[mask, "rsi"].mean()),
                "ATR": float(feat.loc[mask, "atr"].mean()),
                "ClusterName": label,
            }
        )

    # transitions
    trans = np.zeros((k, k), dtype=float)
    cl = feat["cluster"].values
    for i in range(len(cl) - 1):
        trans[cl[i], cl[i + 1]] += 1
    trans = (trans.T / np.clip(trans.sum(axis=1), 1.0, None)).T

    base = pd.DataFrame(
        {
            "Datetime": feat.index.tz_convert("UTC") if getattr(feat.index, "tz", None) is not None else feat.index,
            "Close": feat["Close"].values,
            "ret": feat["ret"].values,
            "vol": feat["vol"].values,
            "rsi": feat["rsi"].values,
            "atr": feat["atr"].values,
            "cluster": feat["cluster"].values,
        }
    )
    base_path = os.path.join(out_dir, f"{tk}_clusters.csv")
    base.to_csv(base_path, index=False)
    stats_path = os.path.join(out_dir, f"{tk}_cluster_stats.csv")
    pd.DataFrame(stats).to_csv(stats_path, index=False)
    trans_path = os.path.join(out_dir, f"{tk}_transitions.csv")
    pd.DataFrame(trans, columns=[f"to_{j}" for j in range(k)]).assign(cluster=np.arange(k)).to_csv(trans_path, index=False)

    fig_paths = _plot_clusters(tk, feat, lab, out_dir, name_map)
    summary = {
        "Tk": tk,
        "k": k,
        "silhouette": sil,
        "n_points": int(len(X)),
    }
    summary_path = os.path.join(out_dir, "summary.csv")
    if os.path.exists(summary_path):
        prev = pd.read_csv(summary_path)
        prev = prev[prev["Tk"] != tk]
        prev = pd.concat([prev, pd.DataFrame([summary])], ignore_index=True)
    else:
        prev = pd.DataFrame([summary])
    prev.to_csv(summary_path, index=False)

    return {
        "base": base_path,
        "stats": stats_path,
        "transitions": trans_path,
        "summary": summary_path,
        **fig_paths,
        "names": name_map,
    }
