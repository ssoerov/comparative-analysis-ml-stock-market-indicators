import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ensure local imports
sys.path.append(os.getcwd())
from vkr_fast.config import Paths, TimeParams, CVParams, TICKERS
from vkr_fast.data import fetch_moex, yahoo_csv, keyrate_series
from vkr_fast.utils.metrics import dm_test
from vkr_fast.mpl_config import configure_matplotlib

configure_matplotlib()


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_aligned_series(paths: Paths, timep: TimeParams, tickers: Dict[str, tuple], offline: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load MOEX for all tickers and external series, align on common timestamps.

    Returns (close_df, exog_df) indexed by UTC time, columns: tickers for close_df; exog: Brent, USD, KeyRate.
    """
    # externals
    brent = yahoo_csv(paths.data_dir, "Brent_yahoo_1h_2023-01-01_2025-04-01 (1).csv")
    usdr = yahoo_csv(paths.data_dir, "USD_RUB_yahoo_1h_2023-01-01_2025-04-01.csv")
    krate = keyrate_series()

    # load per ticker
    frames: Dict[str, pd.DataFrame] = {}
    for tk in tickers:
        df = fetch_moex(
            paths.cache_dir,
            tk,
            *tickers[tk],
            timep.start_raw,
            timep.end_raw,
            timep.interval_minutes,
            use_cache_only=offline,
        )
        frames[tk] = df

    # common index
    idx = None
    for tk, df in frames.items():
        idx = df.index if idx is None else idx.intersection(df.index)
    # limit to VAR/VARMAX period
    period = slice(timep.period_start, timep.period_end - pd.Timedelta(seconds=1))
    idx = pd.DatetimeIndex(idx).tz_convert("UTC")
    idx = idx[(idx >= period.start) & (idx <= period.stop)]

    # build dataframes on common index
    close = pd.DataFrame({tk: frames[tk]["Close"].reindex(idx) for tk in tickers}).dropna()
    exog = pd.DataFrame(index=close.index)
    exog["Brent"] = brent["Close"].reindex(close.index).ffill()
    exog["USD"] = usdr["Close"].reindex(close.index).ffill()
    exog["KeyRate"] = krate["KeyRate"].reindex(close.index).ffill()
    exog.bfill(limit=1, inplace=True)
    return close, exog


def _dclose(df: pd.DataFrame) -> pd.DataFrame:
    out = df.astype(float).diff().dropna()
    return out


def _select_var_order(Y_tr: pd.DataFrame, pmax: int = 6, criterion: str = "bic") -> int:
    import statsmodels.api as sm
    sel = sm.tsa.VAR(Y_tr).select_order(maxlags=pmax)
    c = criterion.lower()
    if hasattr(sel, "selected_orders"):
        # statsmodels >= 0.12
        key = {"aic": "aic", "hq": "hqic", "bic": "bic"}[c]
        p = int(sel.selected_orders.get(key, 1) or 1)
    else:
        if c == "aic":
            val = sel.aic
        elif c == "hq":
            val = sel.hqic
        else:
            val = sel.bic
        if hasattr(val, "idxmin"):
            p = int(val.idxmin())
        else:
            p = int(val)
    return max(1, min(p, pmax))


def _fit_var_results(Y_tr: pd.DataFrame, p: int):
    import statsmodels.api as sm
    model = sm.tsa.VAR(Y_tr)
    res = model.fit(p)
    return res


def _fit_varmax_results(Y_tr: pd.DataFrame, X_tr: pd.DataFrame, p: int):
    import statsmodels.api as sm
    try:
        m = sm.tsa.statespace.VARMAX(Y_tr, order=(p, 0), exog=X_tr, trend='c')
        res = m.fit(disp=False, maxiter=200)
        return res
    except Exception:
        return None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    eps = 1e-8
    e = y_true - y_pred
    mae = float(np.mean(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e**2)))
    wape = float(np.sum(np.abs(e)) / (np.sum(np.abs(y_true)) + eps) * 100)
    smape = float(np.mean(200.0 * np.abs(e) / (np.abs(y_true) + np.abs(y_pred) + eps)))
    mask = np.abs(y_true) > eps
    mape = float(np.mean(np.abs(e[mask] / y_true[mask])) * 100) if mask.any() else float('nan')
    mdape = float(np.median(np.abs(e[mask] / y_true[mask]) * 100)) if mask.any() else float('nan')
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "WAPE": wape, "sMAPE": smape, "MdAPE": mdape}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="VAR/VARMAX walk-forward для IMOEX")
    ap.add_argument("--pmax", type=int, default=6)
    ap.add_argument("--criterion", default="bic", choices=["bic", "aic", "hq"])
    ap.add_argument("--use-varmax", action="store_true", help="Пробовать VARMAX(p,0) с экзогенами")
    ap.add_argument("--exog-lags", type=int, default=0, help="Число лагов для экзогенов (0 = без лагов)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--out-dir", default="outputs/var")
    args = ap.parse_args()

    paths, timep, cvp = Paths(), TimeParams(), CVParams()
    _ensure_dir(args.out_dir)

    # data
    close, exog_levels = _load_aligned_series(paths, timep, TICKERS, offline=args.offline)
    Y = _dclose(close)
    # exog: лог-доходности бенчмарков + уровень ключевой ставки (или её изменений)
    X = pd.DataFrame(index=Y.index)
    X["dBrent"] = np.log(exog_levels["Brent"]).diff().reindex(Y.index)
    X["dUSD"] = np.log(exog_levels["USD"]).diff().reindex(Y.index)
    X["KeyRate"] = exog_levels["KeyRate"].reindex(Y.index)
    X = X.ffill().bfill()
    if args.exog_lags > 0:
        base = list(X.columns)
        for L in range(1, args.exog_lags + 1):
            for c in base:
                X[f"{c}_L{L}"] = X[c].shift(L)
        # выровнять по общим датам
        idx = X.dropna().index.intersection(Y.index)
        X = X.reindex(idx)
        Y = Y.reindex(idx)

    N = len(Y)
    if N <= cvp.test_horizon + args.pmax + 5:
        print("Слишком мало данных для VAR — увеличьте период или уменьшите horizon/pmax.")
        return
    step_raw = (N - cvp.test_horizon) // cvp.outer_folds
    step = max(step_raw, args.pmax + 2)

    orders: List[Tuple[int, int, str, float, int, float]] = []  # (fold, p, model, max_root_mod, is_stable, lm_p)
    rows: List[List] = []  # per-ticker metrics per fold
    rows_agg: List[List] = []  # macro/micro per fold
    dm_rows: List[List] = []  # DM vs SARIMAX and best ML if possible
    dm_best_rows: List[List] = []
    caus_rows: List[List] = []  # Granger causality per fold and direction

    tickers = list(Y.columns)

    for fold in range(cvp.outer_folds):
        start = (fold + 1) * step
        end = start + cvp.test_horizon
        if end > N:
            break
        Y_tr, Y_te = Y.iloc[:start], Y.iloc[start:end]
        X_tr, X_te = X.iloc[:start], X.iloc[start:end]
        # order selection
        p = _select_var_order(Y_tr, pmax=args.pmax, criterion=args.criterion)
        # VAR fit and forecast
        res_var = _fit_var_results(Y_tr, p)
        fc_var = res_var.forecast(Y_tr.values[-p:], steps=len(Y_te))
        # stability & whiteness LM (если доступно)
        try:
            roots = np.asarray(res_var.roots)
            max_root = float(np.max(np.abs(roots))) if roots.size else float("nan")
            is_stable = int(getattr(res_var, "is_stable", lambda: max_root < 1.0)()) if hasattr(res_var, "is_stable") else int(max_root < 1.0)
        except Exception:
            max_root = float("nan")
            is_stable = -1
        try:
            lm = res_var.test_whiteness(nlags=min(12, p + 4))
            lm_p = float(np.nanmin(lm.pvalue)) if hasattr(lm, "pvalue") else float("nan")
        except Exception:
            lm_p = float("nan")
        orders.append((fold, p, "VAR", max_root, is_stable, lm_p))

        # VARMAX optional
        fc_vmx = None
        if args.use_varmax:
            res_vmx = _fit_varmax_results(Y_tr, X_tr, p)
            if res_vmx is not None:
                try:
                    fc_vmx = np.asarray(res_vmx.predict(start=len(Y_tr), end=len(Y_tr)+len(Y_te)-1, exog=X_te))
                except Exception:
                    fc_vmx = None
                # оценки устойчивости для VARMAX: используем условный критерий по корням VAR части если доступно
                max_root_vmx = float("nan")
                is_stable_vmx = -1
                try:
                    if hasattr(res_vmx, "is_stable"):
                        is_stable_vmx = int(res_vmx.is_stable())
                except Exception:
                    pass
                orders.append((fold, p, "VARMAX", max_root_vmx, is_stable_vmx, float("nan")))

        # Granger causality (каждый-на-каждый) для VAR
        try:
            names = list(Y_tr.columns)
            for i, caused in enumerate(names):
                for j, causing in enumerate(names):
                    if i == j:
                        continue
                    try:
                        tst = res_var.test_causality(caused=caused, causing=[causing], kind='f')
                        stat = float(getattr(tst, 'test_statistic', np.nan))
                        pval = float(getattr(tst, 'pvalue', np.nan))
                    except Exception:
                        stat = np.nan; pval = np.nan
                    caus_rows.append([fold, causing, caused, stat, pval])
        except Exception:
            pass

        # IRF и FEVD для VAR (только если модель стабильна)
        try:
            if is_stable == 1:
                irf_h = 10
                irf = res_var.irf(irf_h)
                # График IRF на шок в IMOEX
                if 'IMOEX' in Y_tr.columns:
                    idx_imo = list(Y_tr.columns).index('IMOEX')
                    # Вручную соберём панели
                    import matplotlib.pyplot as plt
                    import matplotlib
                    fig, axes = plt.subplots(nrows=len(Y_tr.columns), ncols=1, figsize=(8, 6))
                    axes = np.atleast_1d(axes)
                    horizon = np.arange(irf_h+1)
                    responses = irf.irfs[:, :, idx_imo]
                    try:
                        lower, upper = irf.cov(bs=1000).irfs_ci(orth=False)  # may not be available in all versions
                    except Exception:
                        lower = upper = None
                    for k, name in enumerate(Y_tr.columns):
                        axes[k].plot(horizon, responses[:, k], label=f"Отклик {name}")
                        if lower is not None and upper is not None:
                            axes[k].fill_between(horizon, lower[:, k, idx_imo], upper[:, k, idx_imo], color='C0', alpha=0.2, label='ДИ')
                        axes[k].grid(True, linestyle='--', alpha=0.5)
                        axes[k].set_ylabel("Отклик")
                    axes[0].set_title("Импульсные отклики на шок в IMOEX (горизонт 10)")
                    axes[-1].set_xlabel("Горизонт (шаги)")
                    fig.tight_layout()
                    fig.savefig(os.path.join(args.out_dir, f"irf_IMOEX_f{fold}.png"))
                    plt.close(fig)
                # FEVD на горизонтах 1 и 5
                fevd = res_var.fevd(5)
                try:
                    decomp = fevd.decomp  # (h, k, k)
                except Exception:
                    decomp = None
                if decomp is not None:
                    for h in (1, 5):
                        mat = decomp[h-1]
                        dfv = pd.DataFrame(mat, index=Y_tr.columns, columns=[f"shock_{c}" for c in Y_tr.columns])
                        dfv.to_csv(os.path.join(args.out_dir, f"fevd_h{h}_f{fold}.csv"))
        except Exception:
            pass

        # metrics per ticker
        for i, tk in enumerate(tickers):
            m_var = _metrics(Y_te.iloc[:, i].values, fc_var[:, i])
            rows.append([fold, tk, "VAR", *m_var.values()])
            if fc_vmx is not None:
                m_vmx = _metrics(Y_te.iloc[:, i].values, fc_vmx[:, i])
                rows.append([fold, tk, "VARMAX", *m_vmx.values()])

        # macro/micro
        # macro: среднее метрик по тикерам
        macro = {}
        micro = {}
        # для компактности рассчитаем сразу на VAR (если есть VARMAX, аналогично)
        Y_true = Y_te.values
        macro_var = _metrics(Y_true, fc_var)
        rows_agg.append([fold, "VAR", *macro_var.values()])
        if fc_vmx is not None:
            macro_vmx = _metrics(Y_true, fc_vmx)
            rows_agg.append([fold, "VARMAX", *macro_vmx.values()])

        # DM vs SARIMAX (если доступны предсказания)
        # загрузим по каждому тикеру outputs/preds/<TK>_f<fold>.csv и сравним на общих датах
        dt_test = Y_te.index
        for i, tk in enumerate(tickers):
            pred_path = os.path.join(paths.out_dir, "preds", f"{tk}_f{fold}.csv")
            if not os.path.exists(pred_path):
                continue
            dfp = pd.read_csv(pred_path)
            dfp["Datetime"] = pd.to_datetime(dfp["Datetime"]).tz_convert("UTC") if dfp["Datetime"].dtype.kind == 'M' else pd.to_datetime(dfp["Datetime"], utc=True)
            dfp = dfp.set_index("Datetime")
            dfp = dfp.reindex(dt_test).dropna()
            if "SARIMAX" in dfp.columns and len(dfp) > 5:
                e_var = (dfp["y_true"].values - fc_var[-len(dfp):, i])
                e_sar = (dfp["y_true"].values - dfp["SARIMAX"].values)
                s, pval = dm_test(e_var, e_sar, h=1, loss="mse")
                dm_rows.append([fold, tk, "VAR", "SARIMAX", float(s), float(pval)])
                if fc_vmx is not None:
                    e_vmx = (dfp["y_true"].values - fc_vmx[-len(dfp):, i])
                    s2, p2 = dm_test(e_vmx, e_sar, h=1, loss="mse")
                    dm_rows.append([fold, tk, "VARMAX", "SARIMAX", float(s2), float(p2)])
            # DM vs best ML/hybrid по RMSE на этих датах
            model_cols = [c for c in dfp.columns if c not in ("y_true", "Close", "Close_prev", "Sigma")]
            if model_cols:
                # выберем модель с минимальным RMSE на текущем окне
                rmses = []
                yv = dfp["y_true"].values
                for mc in model_cols:
                    e = yv - dfp[mc].values
                    rmses.append((mc, float(np.sqrt(np.mean(e**2)))))
                best_name, _ = min(rmses, key=lambda x: x[1])
                e_best = yv - dfp[best_name].values
                if len(dfp) > 5:
                    s3, p3 = dm_test(e_var, e_best, h=1, loss="mse")
                    dm_best_rows.append([fold, tk, "VAR", best_name, float(s3), float(p3)])
                    if fc_vmx is not None:
                        s4, p4 = dm_test(e_vmx, e_best, h=1, loss="mse")
                        dm_best_rows.append([fold, tk, "VARMAX", best_name, float(s4), float(p4)])

    # save outputs
    od = args.out_dir
    _ensure_dir(od)
    cols_m = ["Fold", "Ticker", "Model", "MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "MdAPE"]
    pd.DataFrame(rows, columns=cols_m).to_csv(os.path.join(od, "metrics_per_ticker.csv"), index=False)
    cols_a = ["Fold", "Model", "MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "MdAPE"]
    pd.DataFrame(rows_agg, columns=cols_a).to_csv(os.path.join(od, "metrics_aggregate.csv"), index=False)
    pd.DataFrame(orders, columns=["Fold", "p", "Model", "MaxRootMod", "Stable", "LM_p"]).to_csv(os.path.join(od, "order_selection.csv"), index=False)
    if dm_rows:
        pd.DataFrame(dm_rows, columns=["Fold", "Ticker", "Model", "vs", "DM_stat", "p_val"]).to_csv(
            os.path.join(od, "dm_vs_sarimax.csv"), index=False
        )
    if dm_best_rows:
        pd.DataFrame(dm_best_rows, columns=["Fold", "Ticker", "Model", "vs", "DM_stat", "p_val"]).to_csv(
            os.path.join(od, "dm_vs_best.csv"), index=False
        )
    if caus_rows:
        pd.DataFrame(caus_rows, columns=["Fold", "Cause", "Effect", "stat", "p_val"]).to_csv(
            os.path.join(od, "granger_causality.csv"), index=False
        )
        # агрегируем по значимости p<0.05
        g = pd.DataFrame(caus_rows, columns=["Fold", "Cause", "Effect", "stat", "p_val"]) \
            .assign(sig=lambda d: d["p_val"] < 0.05)
        summ = g.groupby(["Cause", "Effect"]).agg(n_sig=("sig", "sum"), n=("sig", "size")) \
                .assign(share=lambda d: d["n_sig"]/d["n"])
        summ.reset_index().to_csv(os.path.join(od, "granger_causality_summary.csv"), index=False)
    print("VAR/VARMAX расчёты сохранены в outputs/var/")


if __name__ == "__main__":
    main()
