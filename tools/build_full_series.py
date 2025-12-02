import os
import argparse
import sys
sys.path.append(os.getcwd())
from vkr_fast.mpl_config import configure_matplotlib

configure_matplotlib()

import pandas as pd
import matplotlib.pyplot as plt
from vkr_fast.config import Paths, TimeParams, TICKERS
from vkr_fast.data import fetch_moex, yahoo_csv, keyrate_series
from vkr_fast.features import add_indicators, make_lags
from vkr_fast.plotting import apply_gost_style


ALLOWED_MODELS = {"LSTM_att", "CatBoost", "Hybrid"}


def load_raw(paths: Paths, timep: TimeParams):
    # minimal loader mirrored from pipeline
    moex = fetch_moex(
        paths.cache_dir,
        "IMOEX",
        *TICKERS["IMOEX"],
        timep.start_raw,
        timep.end_raw,
        timep.interval_minutes,
        use_cache_only=True,
    )
    brent = yahoo_csv(paths.data_dir, "Brent_yahoo_1h_2023-01-01_2025-04-01 (1).csv")
    usdr = yahoo_csv(paths.data_dir, "USD_RUB_yahoo_1h_2023-01-01_2025-04-01.csv")
    krate = keyrate_series()
    idx = moex.index
    ext = pd.DataFrame(index=idx)
    ext["Brent"] = brent["Close"].reindex(idx).ffill()
    ext["USD"] = usdr["Close"].reindex(idx).ffill()
    ext["KeyRate"] = krate["KeyRate"].reindex(idx).ffill()
    ext.bfill(limit=1, inplace=True)
    return moex.join(ext, how="left")


def stitch_full_series(out_dir: str, target: str):
    paths = Paths(out_dir=out_dir)
    timep = TimeParams()
    raw = load_raw(paths, timep)
    feats = make_lags(add_indicators(raw), window=timep.window, exog_lags=24, target=target)
    base = feats[["Datetime", "y"]].copy()
    base["Datetime"] = pd.to_datetime(base["Datetime"], utc=True)
    pred_dir = os.path.join(out_dir, "preds")
    out_full_dir = os.path.join(out_dir, "full_preds")
    os.makedirs(out_full_dir, exist_ok=True)
    plot_dir = os.path.join(out_dir, "reports_full")
    os.makedirs(plot_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".csv") and "_f" in f)
    for pf in files:
        tk = pf.split("_f")[0]
        fold = pf.split("_f")[1].split(".")[0]
        dfp = pd.read_csv(os.path.join(pred_dir, pf))
        dfp["Datetime"] = pd.to_datetime(dfp["Datetime"], utc=True)
        # Allowed models only
        model_cols = [c for c in dfp.columns if c in ALLOWED_MODELS]
        merged = base.copy()
        merged = merged.merge(
            dfp[["Datetime", "y_true"] + model_cols],
            on="Datetime",
            how="left",
            suffixes=("", ""),
        )
        # keep original target name for clarity
        merged.rename(columns={"y": "y_true_full"}, inplace=True)
        merged_path = os.path.join(out_full_dir, f"{tk}_f{fold}_full.csv")
        merged.to_csv(merged_path, index=False)

        # Plot full series with forecast start marker
        apply_gost_style()
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(merged["Datetime"], merged["y_true_full"], label="Факт", color="black", linewidth=1.5, alpha=0.9)
        # find first forecast timestamp (where any model has non-NaN)
        pred_mask = merged[model_cols].notna().any(axis=1)
        if pred_mask.any():
            fc_start = merged.loc[pred_mask, "Datetime"].min()
            ax.axvline(fc_start, color="#666666", linestyle="--", linewidth=1.2, label="Старт прогноза")
        palette = plt.cm.tab10.colors if len(model_cols) <= 10 else plt.cm.tab20.colors
        for i, c in enumerate(model_cols):
            ax.plot(merged["Datetime"], merged[c], label=c, linewidth=1.8, alpha=0.85, color=palette[i % len(palette)])
        ax.set_xlabel("Дата и время (UTC)")
        ax.set_ylabel("y (ΔЦена / logret)")
        ax.legend(loc="upper left", fontsize=9)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{tk}_f{fold}_full_all_models.png"), bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--target", default="dclose", choices=["dclose", "logret"])
    args = ap.parse_args()
    stitch_full_series(args.out_dir, args.target.lower())


if __name__ == "__main__":
    main()
