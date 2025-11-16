import os
from glob import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ensure local package import
import sys
sys.path.append(os.getcwd())
from vkr_fast.utils.metrics import sharpe_annualized


def econ_from_preds(close: np.ndarray, yhat: np.ndarray, dt: np.ndarray, fee: float, threshold: float, slippage: float) -> Tuple[float, float]:
    log_ret = np.log(close[1:] / close[:-1])
    pos = np.where(yhat > threshold, 1, np.where(yhat < -threshold, -1, 0))
    pos = pos[: len(log_ret)]
    dpos = np.diff(np.insert(pos, 0, 0))
    cost = (fee + slippage) * np.abs(dpos)
    strat = pos * log_ret - cost
    eq = np.exp(np.insert(strat.cumsum(), 0, 0))
    dd = np.maximum.accumulate(eq) - eq
    return eq[-1] - 1, dd.max()


def main():
    ap = argparse.ArgumentParser(description="Optimize no-trade threshold per ticker/model using predictions.")
    ap.add_argument("--fee", type=float, default=0.002)
    ap.add_argument("--slippage", type=float, default=0.0)
    ap.add_argument("--quantiles", default="0,0.5,0.6,0.7,0.8,0.9,0.95", help="Quantiles over |yhat| to form threshold grid")
    ap.add_argument("--out-dir", default="outputs/consolidated")
    ap.add_argument("--split-ratio", type=float, default=0.5, help="Доля валидации внутри каждого файла предсказаний (0..1)")
    args = ap.parse_args()

    pred_files = sorted(glob("outputs/preds/*_f*.csv"))
    if not pred_files:
        print("No prediction files found in outputs/preds")
        return

    # Load predictions grouped by (Tk, Fold)
    groups: Dict[Tuple[str, int], pd.DataFrame] = {}
    for pf in pred_files:
        base = os.path.basename(pf)
        try:
            tk, fold = base.split("_f")
            fold = int(fold.split(".")[0])
        except Exception:
            continue
        df = pd.read_csv(pf)
        groups[(tk, fold)] = df

    # Determine models
    some_df = next(iter(groups.values()))
    base_cols = {"Datetime", "y_true", "Close", "Close_prev", "Sigma"}
    models = [c for c in some_df.columns if c not in base_cols]

    # Build candidate thresholds per (Tk, Model) from quantiles of |yhat| (по валидационной части)
    q_levels = [float(q) for q in args.quantiles.split(",") if q.strip() != ""]
    cand: Dict[Tuple[str, str], List[float]] = {}
    for (tk, _), df in groups.items():
        for m in models:
            n = len(df)
            split = max(1, int(n * args.split_ratio))
            vals = np.abs(df[m].values[:split])
            if len(vals) == 0:
                continue
            qs = np.quantile(vals, q_levels)
            grid = sorted(set([0.0] + [float(x) for x in qs]))
            cand[(tk, m)] = grid

    # Evaluate thresholds by average CumRet across folds (realistic target)
    rows = []
    econ_rows = []
    for tk in sorted({k[0] for k in groups.keys()}):
        tk_folds = sorted([f for (t, f) in groups.keys() if t == tk])
        for m in models:
            grid = cand.get((tk, m))
            if not grid:
                continue
            best_thr = 0.0
            best_cum = -1e18
            for thr in grid:
                cr_list = []
                for f in tk_folds:
                    df = groups[(tk, f)]
                    n = len(df)
                    split = max(1, int(n * args.split_ratio))
                    df_tr = df.iloc[:split]
                    close_curr = df_tr["Close"].values
                    if "Close_prev" in df.columns and len(df["Close_prev"].values) == len(close_curr):
                        close_seq = np.concatenate(([df_tr["Close_prev"].values[0]], close_curr))
                    else:
                        close_seq = np.concatenate(([close_curr[0]], close_curr))
                    yhat = df_tr[m].values
                    dt = pd.to_datetime(df_tr["Datetime"]).values
                    cr, _ = econ_from_preds(close_seq, yhat, dt, args.fee, thr, args.slippage)
                    if not np.isnan(cr):
                        cr_list.append(cr)
                mean_cr = np.mean(cr_list) if cr_list else float("nan")
                if not np.isnan(mean_cr) and mean_cr > best_cum:
                    best_cum = mean_cr
                    best_thr = float(thr)
            rows.append([tk, m, best_thr, best_cum])

            # Save per-fold economics at best threshold
            for f in tk_folds:
                df = groups[(tk, f)]
                n = len(df)
                split = max(1, int(n * args.split_ratio))
                df_te = df.iloc[split:]
                close_curr = df_te["Close"].values
                if "Close_prev" in df.columns and len(df["Close_prev"].values) == len(close_curr):
                    close_seq = np.concatenate(([df_te["Close_prev"].values[0]], close_curr))
                else:
                    close_seq = np.concatenate(([close_curr[0]], close_curr))
                yhat = df_te[m].values
                dt = pd.to_datetime(df_te["Datetime"]).values
                cumret, maxdd = econ_from_preds(close_seq, yhat, dt, args.fee, best_thr, args.slippage)
                econ_rows.append([tk, f, m, best_thr, cumret, maxdd])

    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(rows, columns=["Tk", "Model", "BestThreshold", "CumRet_mean_val"]).to_csv(
        os.path.join(args.out_dir, "thresholds_holdout.csv"), index=False
    )
    pd.DataFrame(econ_rows, columns=["Tk", "Fold", "Model", "Threshold", "CumRet", "MaxDD"]).to_csv(
        os.path.join(args.out_dir, "economics_opt_holdout.csv"), index=False
    )
    print("Threshold optimization (holdout) complete: thresholds_holdout.csv, economics_opt_holdout.csv")


if __name__ == "__main__":
    main()
