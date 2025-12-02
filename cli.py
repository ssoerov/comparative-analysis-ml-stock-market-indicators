#!/usr/bin/env python3
import argparse
from datetime import datetime

from vkr_fast.mpl_config import configure_matplotlib

configure_matplotlib()

from vkr_fast.config import Paths, TimeParams, CVParams, TradingParams, LSTMParams
from vkr_fast.pipeline import run_pipeline, validate_outputs


def _parse_dt(s: str):
    # Expect YYYY-MM-DD
    return datetime.strptime(s, "%Y-%m-%d")


def main():
    p = argparse.ArgumentParser(description="Run VKR Fast pipeline")
    p.add_argument("--data-dir", default="data_input")
    p.add_argument("--cache-dir", default="data_cache")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--model-dir", default="saved_models")

    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--window", type=int, default=60)
    p.add_argument("--start-raw", type=_parse_dt, default=None)
    p.add_argument("--end-raw", type=_parse_dt, default=None)
    p.add_argument("--period-start", type=_parse_dt, default=None)
    p.add_argument("--period-end", type=_parse_dt, default=None)

    p.add_argument("--outer-folds", type=int, default=5)
    p.add_argument("--test-horizon", type=int, default=120)
    p.add_argument("--fee", type=float, default=0.002)
    p.add_argument("--slippage", type=float, default=0.0, help="Per-trade slippage cost (as fraction, applied on position changes)")
    p.add_argument("--trade-threshold", type=float, default=0.0, help="No-trade threshold for signal (abs(signal) <= thr -> flat)")

    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--season-lag", type=int, default=24, help="Сезонный лаг для наивного сезонного прогноза (в барах)")
    p.add_argument("--exog-lags", type=int, default=24, help="Число лагов для экзогенных факторов (Brent, USD, KeyRate)")
    p.add_argument("--embargo-bars", type=int, default=0, help="Число баров-эмбарго между train и test (purged WF)")

    p.add_argument("--offline", action="store_true", help="Use cache-only for MOEX fetch")
    p.add_argument("--no-tf", action="store_true", help="Disable TensorFlow models (LSTM, Hybrid)")
    p.add_argument("--no-catboost", action="store_true", help="Disable CatBoost model")
    p.add_argument("--tickers", default="IMOEX", help="Comma-separated list of tickеров (только IMOEX поддерживается)")
    p.add_argument("--validate", action="store_true", help="Validate outputs after run")
    p.add_argument("--target", default="dclose", choices=["dclose", "logret"], help="Целевая переменная: dclose=приращение цены, logret=лог-доходность")

    # GARCH options
    p.add_argument("--use-garch", action="store_true", help="Fit GARCH on SARIMAX residuals and use sigma")
    p.add_argument("--garch-spec", default="GJR-GARCH", choices=["GARCH","GJR-GARCH","EGARCH"], help="Volatility spec")
    p.add_argument("--garch-dist", default="t", choices=["t","normal"], help="Innovation distribution")
    p.add_argument("--garch-mode", default="feature", choices=["none","feature"], help="How to inject sigma")
    p.add_argument("--risk-scaling", default="none", choices=["none","vol"], help="Scale positions by volatility")
    p.add_argument("--risk-target", type=float, default=1.0, help="Target scale for risk-scaling (units of dClose)")

    args = p.parse_args()

    out_dir = args.out_dir
    if args.target.lower() != "dclose":
        out_dir = f"{out_dir}_{args.target.lower()}"
    paths = Paths(data_dir=args.data_dir, cache_dir=args.cache_dir, out_dir=out_dir, model_dir=args.model_dir)

    def _tz(dt):
        import pandas as pd
        return pd.Timestamp(dt, tz="UTC")

    timep = TimeParams(
        interval_minutes=args.interval,
        window=args.window,
        start_raw=_tz(args.start_raw) if args.start_raw else TimeParams.start_raw,
        end_raw=_tz(args.end_raw) if args.end_raw else TimeParams.end_raw,
        period_start=_tz(args.period_start) if args.period_start else TimeParams.period_start,
        period_end=_tz(args.period_end) if args.period_end else TimeParams.period_end,
    )

    cvp = CVParams(outer_folds=args.outer_folds, test_horizon=args.test_horizon)
    trade = TradingParams(fee=args.fee, slippage=args.slippage, threshold=args.trade_threshold)
    lstm = LSTMParams(epochs=args.epochs, batch_size=args.batch_size)

    # Resolve tickers mapping subset
    from vkr_fast.config import TICKERS
    chosen = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    tickers_map = {t: TICKERS[t] for t in chosen if t in TICKERS}

    run_pipeline(
        paths=paths,
        timep=timep,
        cvp=cvp,
        trade=trade,
        lstm_params=lstm,
        offline=args.offline,
        use_tf=not args.no_tf,
        use_catboost=not args.no_catboost,
        tickers=tickers_map,
        use_garch=args.use_garch,
        garch_spec=args.garch_spec,
        garch_dist=args.garch_dist,
        garch_mode=args.garch_mode,
        risk_scaling=args.risk_scaling,
        risk_target=args.risk_target,
        season_lag=args.season_lag,
        embargo_bars=args.embargo_bars,
        exog_lags=args.exog_lags,
        target=args.target.lower(),
    )

    if args.validate:
        res = validate_outputs(paths)
        if res["ok"]:
            print("Validation: OK")
        else:
            print("Validation: Issues found")
            for issue in res["issues"]:
                print("-", issue)


if __name__ == "__main__":
    main()
