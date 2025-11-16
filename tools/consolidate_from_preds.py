import os
import re
import math
from glob import glob
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE

# ensure local package import
import sys
sys.path.append(os.getcwd())
from vkr_fast.utils.metrics import dm_test

import argparse



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
    args = ap.parse_args()
    pred_files = sorted(glob('outputs/preds/*_f*.csv'))
    if not pred_files:
        print('No prediction files found in outputs/preds')
        return

    met_rows: List[List] = []
    eco_rows: List[List] = []
    err_bucket: Dict[tuple, List[float]] = {}

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
        # Relative MASE denominator from sNaive baseline (if present)
        mase_denom = None
        if 'sNaive' in df.columns:
            try:
                mase_denom = float(MAE(y, df['sNaive'].values))
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
            if mase_denom is not None and np.isfinite(mase_denom):
                mase = float(mae / mase_denom)
            else:
                mase = float('nan')
            met_rows.append([tk, fold, mdl, mae, rmse, mape, wape, smape, mdape, mase])
            cumret, maxdd = econ_from_preds(close_seq, yhat, dt, fee=args.fee, threshold=args.threshold, slippage=args.slippage)
            eco_rows.append([tk, fold, mdl, cumret, maxdd])
            err_bucket.setdefault((tk, mdl), []).extend((y - yhat).tolist())

    # Save consolidated metrics
    mdf = pd.DataFrame(met_rows, columns=["Tk","Fold","Model","MAE","RMSE","MAPE","WAPE","sMAPE","MdAPE","MASE"])
    edf = pd.DataFrame(eco_rows, columns=["Tk","Fold","Model","CumRet","MaxDD"])
    os.makedirs('outputs/consolidated', exist_ok=True)
    mdf.to_csv('outputs/consolidated/metrics_all.csv', index=False)
    edf.to_csv('outputs/consolidated/economics_all.csv', index=False)

    # Sharpe CI removed per request

    # Pairwise DM tests across all models per ticker on aggregated errors
    pair_rows = []
    by_tk: Dict[str, Dict[str, np.ndarray]] = {}
    for (tk, mdl), errs in err_bucket.items():
        by_tk.setdefault(tk, {})[mdl] = np.asarray(errs)
    for tk, mdict in by_tk.items():
        models = sorted(mdict.keys())
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                m1, m2 = models[i], models[j]
                e1, e2 = mdict[m1], mdict[m2]
                L = min(len(e1), len(e2))
                if L < 5:
                    continue
                s, p = dm_test(e1[:L], e2[:L])
                pair_rows.append([tk, m1, m2, s, p])
    if pair_rows:
        pd.DataFrame(pair_rows, columns=["Tk","Model1","Model2","DM_stat","p_val"]).to_csv(
            'outputs/consolidated/dm_test_pairs_all.csv', index=False
        )

    print('Consolidation complete: outputs/consolidated/*.csv')


if __name__ == '__main__':
    main()
