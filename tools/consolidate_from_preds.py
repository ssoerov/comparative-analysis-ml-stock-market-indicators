import os
import re
import math
from glob import glob
from typing import List, Dict
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE

# ensure local package import
import sys
sys.path.append(os.getcwd())
from vkr_fast.utils.metrics import dm_test

import argparse


DM_METRICS = ("mae", "rmse", "mape", "wape", "smape", "mdape")


def _per_point_losses(y_true, y_pred, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    abs_err = np.abs(diff)
    denom = np.abs(y_true)
    mask = denom > eps
    return {
        "mae": abs_err,
        "rmse": diff ** 2,
        "mape": np.where(mask, abs_err / denom * 100.0, np.nan),
        "wape": np.where(mask, abs_err / denom * 100.0, np.nan),
        "smape": 200.0 * abs_err / (np.abs(y_true) + np.abs(y_pred) + eps),
        "mdape": np.where(mask, abs_err / denom * 100.0, np.nan),
    }


def _align_losses(l1: np.ndarray, l2: np.ndarray):
    a1 = np.asarray(l1, dtype=float).ravel()
    a2 = np.asarray(l2, dtype=float).ravel()
    L = min(len(a1), len(a2))
    if L == 0:
        return np.array([]), np.array([])
    a1 = a1[:L]
    a2 = a2[:L]
    mask = np.isfinite(a1) & np.isfinite(a2)
    return a1[mask], a2[mask]



def econ_from_preds(close: np.ndarray, yhat: np.ndarray, dt: np.ndarray, fee: float = 0.002, threshold: float = 0.0, slippage: float = 0.0):
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--fee', type=float, default=0.002)
    ap.add_argument('--slippage', type=float, default=0.0)
    ap.add_argument('--threshold', type=float, default=0.0)
    ap.add_argument('--out-dir', default='outputs')
    args = ap.parse_args()
    out_dir = args.out_dir
    pred_files = sorted(glob(f"{out_dir}/preds/*_f*.csv"))
    if not pred_files:
        print(f'No prediction files found in {out_dir}/preds')
        return

    met_rows: List[List] = []
    eco_rows: List[List] = []
    loss_bucket = defaultdict(lambda: {m: [] for m in DM_METRICS})

    pat = re.compile(r'.*/([A-Z]+)_f(\d+)\.csv$')
    for pf in pred_files:
        m = pat.match(pf)
        if not m:
            continue
        tk, fold = m.group(1), int(m.group(2))
        df = pd.read_csv(pf)
        base_cols = {'Datetime', 'y_true', 'Close', 'Close_prev', 'Sigma'}
        models = [c for c in df.columns if c not in base_cols]
        y = df['y_true'].values
        close_curr = df['Close'].values
        if 'Close_prev' in df.columns and len(df['Close_prev'].values) == len(close_curr):
            close_seq = np.concatenate(([df['Close_prev'].values[0]], close_curr))
        else:
            close_seq = np.concatenate(([close_curr[0]], close_curr))
        dt = pd.to_datetime(df['Datetime']).values
        # MASE denominator: MAE одношагового наивного прогноза (lag1) на этой выборке
        mase_denom = None
        if 'Naive' in df.columns:
            try:
                mase_denom = float(MAE(y, df['Naive'].values))
                if mase_denom == 0:
                    mase_denom = None
            except Exception:
                mase_denom = None
        for mdl in models:
            yhat = df[mdl].values
            eps = 1e-8
            mae = MAE(y, yhat)
            rmse = math.sqrt(MSE(y, yhat))
            mask = np.abs(y) > eps
            mape = float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100) if mask.any() else float("nan")
            wape = float(np.sum(np.abs(y - yhat)) / (np.sum(np.abs(y)) + eps) * 100)
            smape = float(np.mean(200.0 * np.abs(y - yhat) / (np.abs(y) + np.abs(yhat) + eps)))
            mdape = float(np.median(np.abs((y[mask] - yhat[mask]) / y[mask]) * 100)) if mask.any() else float("nan")
            mase = float(mae / mase_denom) if (mase_denom is not None and np.isfinite(mase_denom) and mase_denom > 0) else float('nan')
            met_rows.append([tk, fold, mdl, mae, rmse, mape, wape, smape, mdape, mase])
            cumret, maxdd = econ_from_preds(close_seq, yhat, dt, fee=args.fee, threshold=args.threshold, slippage=args.slippage)
            eco_rows.append([tk, fold, mdl, cumret, maxdd])
            losses = _per_point_losses(y, yhat, eps=eps)
            for metric_key, series in losses.items():
                loss_bucket[(tk, mdl)][metric_key].extend(series.tolist())

    # Save consolidated metrics
    mdf = pd.DataFrame(met_rows, columns=["Tk","Fold","Model","MAE","RMSE","MAPE","WAPE","sMAPE","MdAPE","MASE"])
    edf = pd.DataFrame(eco_rows, columns=["Tk","Fold","Model","CumRet","MaxDD"])
    cons_dir = os.path.join(out_dir, "consolidated")
    os.makedirs(cons_dir, exist_ok=True)
    mdf.to_csv(os.path.join(cons_dir, "metrics_all.csv"), index=False)
    edf.to_csv(os.path.join(cons_dir, "economics_all.csv"), index=False)

    # Sharpe CI removed per request

    # Pairwise DM tests across all models per ticker and per metric on aggregated losses
    pair_rows = []
    by_tk: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for (tk, mdl), losses in loss_bucket.items():
        by_tk.setdefault(tk, {})[mdl] = losses

    for tk, mdict in by_tk.items():
        models = sorted(mdict.keys())
        for metric_key in DM_METRICS:
            for i in range(len(models)):
                for j in range(i+1, len(models)):
                    m1, m2 = models[i], models[j]
                    l1 = mdict[m1].get(metric_key, [])
                    l2 = mdict[m2].get(metric_key, [])
                    l1, l2 = _align_losses(l1, l2)
                    if len(l1) < 5:
                        continue
                    s, p = dm_test(l1, l2, input_is_loss=True)
                    pair_rows.append([tk, m1, m2, metric_key.upper(), s, p])
    if pair_rows:
        pd.DataFrame(pair_rows, columns=["Tk","Model1","Model2","Metric","DM_stat","p_val"]).to_csv(
            os.path.join(cons_dir, "dm_test_pairs_metrics.csv"), index=False
        )

    print(f'Consolidation complete: {cons_dir}/*.csv')


if __name__ == '__main__':
    main()
