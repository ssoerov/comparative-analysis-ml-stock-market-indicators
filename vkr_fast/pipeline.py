import json
import math
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE, mean_squared_error as MSE

from .mpl_config import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
from .config import (
    Paths,
    TimeParams,
    CVParams,
    TradingParams,
    LSTMParams,
    TICKERS,
    ensure_dirs,
    setup_environment,
    cpu_count_limited,
    default_logger,
)
from .data import yahoo_csv, keyrate_series, fetch_moex
from .features import add_indicators, make_lags
from .models import (
    fit_sarimax,
    sarimax_predict,
    fit_random_forest,
    fit_catboost,
)
from .models.garch import fit_garch, forecast_garch
from .utils import dm_test, block_bootstrap_means
from .reporting import generate_reports
from .analysis import chain_growth_report, cluster_regimes_for_ticker

DM_METRICS = ("mae", "rmse", "mape", "wape", "smape", "mdape")


def _per_point_losses(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> Dict[str, np.ndarray]:
    """Compute per-point losses for metrics used in DM tests."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    abs_err = np.abs(diff)
    denom = np.abs(y_true)
    mask = denom > eps
    losses = {
        "mae": abs_err,
        "rmse": diff ** 2,
        "mape": np.where(mask, abs_err / denom * 100.0, np.nan),
        "wape": np.where(mask, abs_err / denom * 100.0, np.nan),
        "smape": 200.0 * abs_err / (np.abs(y_true) + np.abs(y_pred) + eps),
        "mdape": np.where(mask, abs_err / denom * 100.0, np.nan),
    }
    return losses


def _align_losses(l1: np.ndarray, l2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align two loss series to equal length and drop non-finite pairs."""
    a1 = np.asarray(l1, dtype=float).ravel()
    a2 = np.asarray(l2, dtype=float).ravel()
    L = min(len(a1), len(a2))
    if L == 0:
        return np.array([]), np.array([])
    a1 = a1[:L]
    a2 = a2[:L]
    mask = np.isfinite(a1) & np.isfinite(a2)
    return a1[mask], a2[mask]


def _load_all_data(paths: Paths, timep: TimeParams, cpu: int, logger, offline: bool = False, tickers=None) -> Dict[str, pd.DataFrame]:
    logger.info("Загружаем MOEX – Brent – USD/RUB – KeyRate ...")
    t0 = pd.Timestamp.utcnow()
    tickers = tickers or TICKERS
    if offline:
        # Cache-only path
        moex = {}
        for tk in tickers:
            moex[tk] = fetch_moex(
                paths.cache_dir,
                tk,
                *tickers[tk],
                timep.start_raw,
                timep.end_raw,
                timep.interval_minutes,
                use_cache_only=True,
            )
    else:
        with ThreadPoolExecutor(cpu) as pool:
            futures = {
                pool.submit(
                    fetch_moex,
                    paths.cache_dir,
                    tk,
                    *tickers[tk],
                    timep.start_raw,
                    timep.end_raw,
                    timep.interval_minutes,
                    False,
                ): tk
                for tk in tickers
            }
            moex = {tk: fut.result() for fut, tk in futures.items()}

    brent = yahoo_csv(paths.data_dir, "Brent_yahoo_1h_2023-01-01_2025-04-01 (1).csv")
    usdr = yahoo_csv(paths.data_dir, "USD_RUB_yahoo_1h_2023-01-01_2025-04-01.csv")
    krate = keyrate_series()
    logger.info("✓ Данные загружены за %.1f сек.", (pd.Timestamp.utcnow() - t0).total_seconds())

    raw = {}
    period_slice = slice(timep.period_start, timep.period_end - pd.Timedelta(seconds=1))
    for tk in tickers:
        base = moex[tk]
        idx = base.index
        ext = pd.DataFrame(index=idx)
        ext["Brent"] = brent["Close"].reindex(idx).ffill()
        ext["USD"] = usdr["Close"].reindex(idx).ffill()
        ext["KeyRate"] = krate["KeyRate"].reindex(idx).ffill()
        ext.bfill(limit=1, inplace=True)
        joined = base.join(ext, how="left")
        raw[tk] = joined.loc[period_slice]
    return raw


def _build_features(raw: Dict[str, pd.DataFrame], cpu: int, win: int, exog_lags: int) -> Dict[str, pd.DataFrame]:
    feats = {tk: make_lags(add_indicators(raw[tk], n_jobs=cpu), window=win, exog_lags=exog_lags) for tk in raw}
    return feats


def _build_sequences(X: np.ndarray, window: int) -> np.ndarray:
    return np.stack([X[i - window : i] for i in range(window, len(X))]).astype("float32")


def run_pipeline(
    paths: Paths = Paths(),
    timep: TimeParams = TimeParams(),
    cvp: CVParams = CVParams(),
    trade: TradingParams = TradingParams(),
    lstm_params: LSTMParams = LSTMParams(),
    offline: bool = False,
    use_tf: bool = True,
    use_catboost: bool = True,
    tickers: Dict[str, tuple] = None,
    use_garch: bool = False,
    garch_spec: str = "GJR-GARCH",
    garch_dist: str = "t",
    garch_mode: str = "feature",  # or "none"
    risk_scaling: str = "none",    # or "vol"
    risk_target: float = 1.0,
    season_lag: int = 24,
    embargo_bars: int = 0,
    val_frac: float = 0.15,
    exog_lags: int = 24,
) -> None:
    ensure_dirs(paths)
    setup_environment()
    plt.rcParams.update({"figure.dpi": 120})
    cpu = cpu_count_limited(4)
    logger = default_logger()

    # Optional LSTM readiness check
    if use_tf:
        try:
            import tensorflow as _tf  # noqa: F401
        except Exception:
            logger.warning("TensorFlow не найден — LSTM/Hybrid будут отключены. Установите TF и перезапустите (или добавьте --no-tf).")
            use_tf = False

    # Load data
    tmap = tickers or TICKERS
    raw = _load_all_data(paths, timep, cpu, logger, offline=offline, tickers=tmap)

    # Features
    feats = _build_features(raw, cpu, timep.window, exog_lags=exog_lags)
    sample_df = next(iter(feats.values()))
    base_cols = {"Datetime", "dClose", "Close", "Open", "High", "Low", "Volume"}
    feature_manifest = [c for c in sample_df.columns if c not in base_cols]
    with open(os.path.join(paths.out_dir, "feature_space.json"), "w", encoding="utf-8") as fh:
        json.dump({"target": "dClose", "features": feature_manifest}, fh, ensure_ascii=False, indent=2)
    analysis_dir = os.path.join(paths.out_dir, "imoex_analysis")
    analysis_done = set()

    # Walk-forward
    metrics_rows: List[List] = []
    econ_rows: List[List] = []
    err_rows: List[List] = []
    feature_rows: List[List] = []
    loss_bucket = defaultdict(lambda: {m: [] for m in DM_METRICS})

    for tk in tmap:
        logger.info("================  %s  ================", tk)
        if tk not in analysis_done:
            try:
                chain_growth_report(raw[tk]["Close"], tk, analysis_dir)
            except Exception as exc:
                logger.warning("[%s] Цепной анализ не построен: %s", tk, exc)
            try:
                cluster_regimes_for_ticker(tk, raw[tk], os.path.join(paths.out_dir, "clustering"))
            except Exception as exc:
                logger.warning("[%s] Кластеризация не выполнена: %s", tk, exc)
            analysis_done.add(tk)
        df = feats[tk]
        history = len(df)
        if history <= timep.window + cvp.test_horizon:
            logger.warning(
                "[%s] Недостаточно данных для %d фолдов (history=%d, window=%d, horizon=%d) — тикер пропущен.",
                tk,
                cvp.outer_folds,
                history,
                timep.window,
                cvp.test_horizon,
            )
            continue

        raw_step = max(1, (history - cvp.test_horizon) // cvp.outer_folds)
        step = raw_step  # стараемся покрыть все фолды, контролируем минимум train через проверки ниже

        for fold in range(cvp.outer_folds):
            start = (fold + 1) * step
            end = start + cvp.test_horizon
            if end > history:
                logger.warning("[%s] fold %d выходит за пределы выборки (end=%d, history=%d) — прекращаем цикл.", tk, fold, end, history)
                break

            logger.info("⮑  fold %d/%d", fold + 1, cvp.outer_folds)
            tr = df.iloc[:start]
            te = df.iloc[start:end]

            if embargo_bars > 0 and len(tr) > embargo_bars:
                tr = tr.iloc[:-embargo_bars]

            if len(tr) <= timep.window:
                logger.warning(
                    "[%s] fold %d: длина train=%d <= window=%d — пропускаем фолд.",
                    tk,
                    fold,
                    len(tr),
                    timep.window,
                )
                continue

            if len(te) <= timep.window:
                logger.warning(
                    "[%s] fold %d: длина test=%d <= window=%d — пропускаем фолд.",
                    tk,
                    fold,
                    len(te),
                    timep.window,
                )
                continue

            y_tr = tr["dClose"].values.astype("float32")
            y_te = te["dClose"].values.astype("float32")
            from pandas.api.types import is_numeric_dtype
            feat_cols = [c for c in tr.columns if c != "dClose" and is_numeric_dtype(tr[c])]
            X_tr = tr[feat_cols].values.astype("float32")
            X_te = te[feat_cols].values.astype("float32")

            xsc = StandardScaler().fit(X_tr)
            ysc = StandardScaler().fit(y_tr.reshape(-1, 1))
            X_trs, X_tes = xsc.transform(X_tr), xsc.transform(X_te)
            y_trz = ysc.transform(y_tr.reshape(-1, 1)).ravel()
            y_tez = ysc.transform(y_te.reshape(-1, 1)).ravel()

            # SARIMAX (fit early; needed for GARCH residuals and hybrid)
            sar = fit_sarimax(y_tr, X_trs)
            p_sar = sarimax_predict(sar, start=len(y_tr), end=len(y_tr) + len(y_te) - 1, exog_future=X_tes)

            # GARCH on SARIMAX residuals (optional)
            sigma_tr = sigma_te = None
            if use_garch:
                mu_tr = sarimax_predict(sar, start=0, end=len(y_tr) - 1, exog_future=X_trs)
                e_tr = y_tr - mu_tr
                try:
                    gfit = fit_garch(e_tr, vol=garch_spec, dist=garch_dist)
                    # in-sample sigma for train (aligned)
                    sigma_tr_ins = np.sqrt(np.maximum(gfit.conditional_volatility, 0.0))
                    # forecast sigma for test horizon
                    _, sigma_te = forecast_garch(gfit, horizon=len(y_te))
                    sigma_tr = np.asarray(sigma_tr_ins, dtype=float)
                except Exception:
                    sigma_tr = sigma_te = None

            # Inject sigma as feature (if available and requested)
            X_trs_used = X_trs
            X_tes_used = X_tes
            feat_names_used = list(feat_cols)
            if use_garch and garch_mode == "feature" and (sigma_tr is not None) and (sigma_te is not None):
                X_trs_used = np.hstack([X_trs, sigma_tr[:, None]])
                X_tes_used = np.hstack([X_tes, sigma_te[:, None]])
                feat_names_used = feat_names_used + ["Sigma"]

            # ML models
            rf = fit_random_forest(X_trs_used, y_tr)
            p_rf = rf.predict(X_tes_used)
            if hasattr(rf, "feature_importances_"):
                rf_imp = getattr(rf, "feature_importances_", None)
                if rf_imp is not None:
                    total_imp = float(np.sum(rf_imp))
                    if total_imp > 0:
                        for fname, val in zip(feat_names_used, rf_imp):
                            feature_rows.append([tk, fold, "RF", fname, float(val / total_imp)])

            if use_catboost:
                cb = fit_catboost(X_trs_used, y_tr)
                p_cb = cb.predict(X_tes_used)
                try:
                    cb_imp = cb.get_feature_importance()
                    total_cb = float(np.sum(cb_imp))
                    if total_cb > 0:
                        for fname, val in zip(feat_names_used, cb_imp):
                            feature_rows.append([tk, fold, "CatBoost", fname, float(val / total_cb)])
                except Exception:
                    pass
            else:
                p_cb = None

            # LSTM sequences (lazy import to avoid TF unless requested)
            win = timep.window
            if use_tf:
                from .models.lstm import lstm_arch, fit_lstm

                seq_tr = _build_sequences(X_trs_used, win)
                seq_te = _build_sequences(X_tes_used, win)
                y_tr_seq = y_trz[win:]
                y_te_seq = y_tez[win:]

                # Validation split inside train (time-based) — no leakage to test
                n_tr = len(seq_tr)
                n_val = max(1, int(n_tr * max(0.05, min(0.4, val_frac))))
                X_tr_lstm, y_tr_lstm = seq_tr[:-n_val], y_tr_seq[:-n_val]
                X_val_lstm, y_val_lstm = seq_tr[-n_val:], y_tr_seq[-n_val:]

                lb = fit_lstm(
                    lstm_arch(seq_tr.shape[1:], with_attention=False),
                    X_tr_lstm,
                    y_tr_lstm,
                    X_val_lstm,
                    y_val_lstm,
                    epochs=lstm_params.epochs,
                    batch_size=lstm_params.batch_size,
                )
                p_lb = ysc.inverse_transform(lb.predict(seq_te).astype("float32")).ravel()

                la = fit_lstm(
                    lstm_arch(seq_tr.shape[1:], with_attention=True),
                    X_tr_lstm,
                    y_tr_lstm,
                    X_val_lstm,
                    y_val_lstm,
                    epochs=lstm_params.epochs,
                    batch_size=lstm_params.batch_size,
                )
                p_la = ysc.inverse_transform(la.predict(seq_te).astype("float32")).ravel()

                # Hybrid LSTM+SARIMAX feature (+ GARCH sigma если доступно)
                extra_tr = [sar.predict()[-len(X_tr) : , None]]
                extra_te = [p_sar[:, None]]
                if use_garch and (sigma_tr is not None) and (sigma_te is not None):
                    extra_tr.append(sigma_tr[:, None])
                    extra_te.append(sigma_te[:, None])
                Xtr_h = np.hstack([X_trs_used, *extra_tr])
                Xte_h = np.hstack([X_tes_used, *extra_te])
                seq_tr_h = _build_sequences(Xtr_h, win)
                seq_te_h = _build_sequences(Xte_h, win)

                # Use same validation split size as for base/att models
                n_tr_h = len(seq_tr_h)
                n_val_h = min(n_val, n_tr_h - 1)
                X_tr_h_lstm, y_tr_h_lstm = seq_tr_h[:-n_val_h], y_tr_seq[:-n_val_h]
                X_val_h_lstm, y_val_h_lstm = seq_tr_h[-n_val_h:], y_tr_seq[-n_val_h:]
                lh = fit_lstm(
                    lstm_arch(seq_tr_h.shape[1:], with_attention=True),
                    X_tr_h_lstm,
                    y_tr_h_lstm,
                    X_val_h_lstm,
                    y_val_h_lstm,
                    epochs=lstm_params.epochs,
                    batch_size=lstm_params.batch_size,
                )
                p_h = ysc.inverse_transform(lh.predict(seq_te_h).astype("float32")).ravel()
                lh.save(f"{paths.model_dir}/{tk}_f{fold}_hybrid.keras")
            else:
                p_lb = p_la = p_h = None

            # Metrics
            cut = timep.window
            y_eval = y_te[cut:]
            preds = {"RF": p_rf[cut:], "SARIMAX": p_sar[cut:]}
            # Naive baselines
            naive = np.zeros_like(y_eval)
            y_full = np.concatenate([y_tr, y_te])
            test_start = len(y_tr) + cut
            lag = max(0, int(season_lag))
            if lag > 0:
                idx = np.arange(len(y_eval)) + test_start - lag
                seas = np.where(idx >= 0, y_full[idx], 0.0)
            else:
                seas = np.zeros_like(y_eval)
            preds["Naive"] = naive
            preds["sNaive"] = seas
            if use_catboost and p_cb is not None:
                preds["CatBoost"] = p_cb[cut:]
            if use_tf and p_lb is not None:
                preds.update({"LSTM_base": p_lb, "LSTM_att": p_la, "Hybrid": p_h})

            eps = 1e-8
            mase_denom = float(np.mean(np.abs(y_tr))) + eps
            for name, y_hat in preds.items():
                mae = MAE(y_eval, y_hat)
                rmse = math.sqrt(MSE(y_eval, y_hat))
                mask = np.abs(y_eval) > eps
                mape = float(np.mean(np.abs((y_eval[mask] - y_hat[mask]) / y_eval[mask])) * 100) if mask.any() else float("nan")
                wape = float(np.sum(np.abs(y_eval - y_hat)) / (np.sum(np.abs(y_eval)) + eps) * 100)
                smape = float(np.mean(200.0 * np.abs(y_eval - y_hat) / (np.abs(y_eval) + np.abs(y_hat) + eps)))
                mdape = float(np.median(np.abs((y_eval[mask] - y_hat[mask]) / y_eval[mask]) * 100)) if mask.any() else float("nan")
                mase = float(mae / mase_denom)
                metrics_rows.append([tk, fold, name, mae, rmse, mape, wape, smape, mdape, mase])
                err_rows.extend([[tk, name, e] for e in y_eval - y_hat])
                losses = _per_point_losses(y_eval, y_hat, eps=eps)
                for metric_key, series in losses.items():
                    loss_bucket[(tk, name)][metric_key].extend(series.tolist())

            # Save per-fold predictions for detailed reporting
            pred_dir = f"{paths.out_dir}/preds"
            os.makedirs(pred_dir, exist_ok=True)
            # Восстанавливаем временной столбец из универсального имени
            time_col = None
            for c in ("Datetime", "index", "begin"):
                if c in te.columns:
                    time_col = c
                    break
            if time_col is None:
                # попытка найти datetime-тип
                dt_cols = [c for c in te.columns if str(te[c].dtype).startswith("datetime64")]
                time_col = dt_cols[0] if dt_cols else None
            if time_col is None:
                raise KeyError("Не найден временной столбец (ожидались 'Datetime'/'index'/'begin').")
            dt_index = te[time_col].values[cut:]
            close_curr = te["Close"].values[cut:]
            close_prev = te["Close"].values[cut - 1 : -1]
            pred_df = pd.DataFrame(
                {
                    "Datetime": dt_index,
                    "y_true": y_eval,
                    "Close": close_curr,
                    "Close_prev": close_prev,
                }
            )
            for name, y_hat in preds.items():
                pred_df[name] = y_hat
            # Save sigma (aligned with evaluation slice) if available
            if use_garch and sigma_te is not None:
                sigma_al = sigma_te[cut:]
                if len(sigma_al) == len(pred_df):
                    pred_df["Sigma"] = sigma_al
            pred_path = f"{pred_dir}/{tk}_f{fold}.csv"
            pred_df.to_csv(pred_path, index=False)

            # Economics
            close_eval = te["Close"].values[cut - 1 :]
            log_ret = np.log(close_eval[1:] / close_eval[:-1])
            for name, y_hat in preds.items():
                thr = trade.threshold
                # base binary positions
                pos = np.where(y_hat > thr, 1.0, np.where(y_hat < -thr, -1.0, 0.0))
                # risk scaling by volatility if requested and sigma available
                if risk_scaling == "vol" and use_garch and sigma_te is not None:
                    sigma_al = sigma_te[cut:]
                    if len(sigma_al) == len(pos):
                        scale = np.clip(risk_target / (sigma_al + 1e-8), 0.0, 1.0)
                        pos = np.sign(pos) * np.minimum(np.abs(pos), scale)
                dpos = np.diff(np.insert(pos, 0, 0))
                cost = (trade.fee + trade.slippage) * np.abs(dpos)
                strat = pos * log_ret - cost
                eq = np.exp(np.insert(strat.cumsum(), 0, 0))
                dd = np.maximum.accumulate(eq) - eq
                econ_rows.append([tk, fold, name, eq[-1] - 1, dd.max()])

            logger.info("   … fold готов")

            # Generate detailed GOST-styled reports per fold
            try:
                # История для общего графика: весь train + весь horizon
                hist_dt = np.concatenate([tr[time_col].values, te[time_col].values])
                hist_y = np.concatenate([tr["dClose"].values, te["dClose"].values])
                fc_start_dt = te[time_col].values[cut] if len(te[time_col].values) > cut else hist_dt[-len(y_eval)]
                generate_reports(
                    pred_df,
                    paths.out_dir,
                    tk,
                    fold,
                    trade.fee,
                    threshold=trade.threshold,
                    slippage=trade.slippage,
                    gost=True,
                    history=(pd.to_datetime(hist_dt), hist_y, pd.to_datetime(fc_start_dt)),
                )
            except Exception:
                pass

    # Save outputs
    pd.DataFrame(metrics_rows, columns=["Tk", "Fold", "Model", "MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "MdAPE", "MASE"]).to_csv(
        f"{paths.out_dir}/metrics.csv", index=False
    )
    pd.DataFrame(econ_rows, columns=["Tk", "Fold", "Model", "CumRet", "MaxDD"]).to_csv(
        f"{paths.out_dir}/economics.csv", index=False
    )
    if feature_rows:
        pd.DataFrame(feature_rows, columns=["Tk", "Fold", "Model", "Feature", "Importance"]).to_csv(
            f"{paths.out_dir}/feature_importance.csv", index=False
        )

    err_df = pd.DataFrame(err_rows, columns=["Tk", "Mdl", "err"])
    ci, dm_rows = [], []
    for (tk, mdl), g in err_df.groupby(["Tk", "Mdl"]):
        e = g["err"].values
        # Moving block bootstrap for time series errors (more realistic than i.i.d.)
        bs = block_bootstrap_means(e, B=600)
        ci.append([tk, mdl, *np.percentile(bs, [2.5, 50, 97.5])])
        if mdl != "Hybrid":
            h_losses = loss_bucket.get((tk, "Hybrid"), {})
            if not h_losses:
                continue
            for metric_key in DM_METRICS:
                l1, l2 = _align_losses(loss_bucket[(tk, mdl)].get(metric_key, []), h_losses.get(metric_key, []))
                if len(l1) < 5:
                    continue
                stat, p = dm_test(l1, l2, input_is_loss=True)
                dm_rows.append([tk, mdl, metric_key.upper(), stat, p])

    pd.DataFrame(ci, columns=["Tk", "Mdl", "CI_low", "CI_med", "CI_hi"]).to_csv(
        f"{paths.out_dir}/bootstrap_CI.csv", index=False
    )
    if dm_rows:
        pd.DataFrame(dm_rows, columns=["Tk", "vs", "Metric", "DM_stat", "p_val"]).to_csv(
            f"{paths.out_dir}/dm_test.csv", index=False
        )

    # Pairwise DM tests across all available models per ticker for each metric
    pair_rows = []
    for tk in tmap:
        models = sorted({mdl for (tk_key, mdl) in loss_bucket.keys() if tk_key == tk})
        if not models:
            continue
        for metric_key in DM_METRICS:
            for i in range(len(models)):
                for j in range(i + 1, len(models)):
                    m1, m2 = models[i], models[j]
                    l1 = loss_bucket[(tk, m1)].get(metric_key, [])
                    l2 = loss_bucket[(tk, m2)].get(metric_key, [])
                    l1, l2 = _align_losses(l1, l2)
                    if len(l1) < 5:
                        continue
                    s, p = dm_test(l1, l2, input_is_loss=True)
                    pair_rows.append([tk, m1, m2, metric_key.upper(), s, p])
    if pair_rows:
        pd.DataFrame(pair_rows, columns=["Tk", "Model1", "Model2", "Metric", "DM_stat", "p_val"]).to_csv(
            f"{paths.out_dir}/dm_test_pairs.csv", index=False
        )

    for tk in tmap:
        plt.figure(figsize=(10, 4))
        s = feats[tk].copy()
        # Определяем столбец времени
        time_col = None
        for c in ("Datetime", "index", "begin"):
            if c in s.columns:
                time_col = c
                break
        if time_col is None:
            dt_cols = [c for c in s.columns if str(s[c].dtype).startswith("datetime64")]
            time_col = dt_cols[0] if dt_cols else None
        if time_col is None:
            plt.close()
            continue
        s[time_col] = pd.to_datetime(s[time_col])
        p_start = pd.to_datetime(s[time_col]).min()
        p_end = pd.to_datetime(s[time_col]).max()
        plt.plot(s[time_col], s["Close"], label="Цена закрытия")
        plt.xlabel("Дата и время (UTC)")
        plt.ylabel("Цена закрытия")
        plt.title(f"{tk} — Динамика ряда цены закрытия за период {p_start:%Y-%m-%d} — {p_end:%Y-%m-%d}")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(f"{paths.out_dir}/{tk}_close.png")
        plt.close()

    logger.info("✔ Pipeline complete — результаты сохранены в ./outputs")


def validate_outputs(paths: Paths) -> dict:
    """Basic validation of produced outputs: existence, columns, non-empty."""
    import os

    results = {"ok": True, "issues": []}

    def check_csv(path: str, required_cols: list):
        if not os.path.exists(path):
            results["ok"] = False
            results["issues"].append(f"Missing file: {path}")
            return None
        df = pd.read_csv(path)
        for c in required_cols:
            if c not in df.columns:
                results["ok"] = False
                results["issues"].append(f"{path} missing column: {c}")
        if len(df) == 0:
            results["ok"] = False
            results["issues"].append(f"{path} is empty")
        return df

    m = check_csv(f"{paths.out_dir}/metrics.csv", ["Tk", "Fold", "Model", "MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "MdAPE"])
    e = check_csv(f"{paths.out_dir}/economics.csv", ["Tk", "Fold", "Model", "CumRet", "MaxDD"])
    ci = check_csv(f"{paths.out_dir}/bootstrap_CI.csv", ["Tk", "Mdl", "CI_low", "CI_med", "CI_hi"])
    dm = check_csv(f"{paths.out_dir}/dm_test.csv", ["Tk", "vs", "DM_stat", "p_val"])

    # If no Hybrid model was produced, allow empty/missing DM file
    if m is not None and ("Hybrid" not in set(m["Model"].unique())):
        if dm is None or len(dm) == 0:
            # remove previous issues related to dm_test.csv
            results["issues"] = [iss for iss in results["issues"] if "dm_test.csv" not in iss]
            # keep ok status if no other issues
            if len(results["issues"]) == 0:
                results["ok"] = True

    # Light plausibility checks
    if m is not None:
        if (m[["MAE", "RMSE", "MAPE", "WAPE"]] < 0).any().any():
            results["ok"] = False
            results["issues"].append("Negative error metrics detected")
    if e is not None:
        if (e["MaxDD"] < 0).any():
            # drawdown should be >= 0
            results["ok"] = False
            results["issues"].append("Negative MaxDD detected")

    return results
