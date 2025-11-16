import os
import sys

import pandas as pd

sys.path.append(os.getcwd())
from vkr_fast.mpl_config import configure_matplotlib

configure_matplotlib()

from vkr_fast.config import Paths, TimeParams, TICKERS
from vkr_fast.data import fetch_moex, yahoo_csv, keyrate_series
from vkr_fast.analysis import chain_growth_report, cluster_regimes_for_ticker


def _load_imoex(paths: Paths, timep: TimeParams) -> pd.DataFrame:
    tk = next(iter(TICKERS.keys()))
    df = fetch_moex(
        paths.cache_dir,
        tk,
        *TICKERS[tk],
        timep.start_raw,
        timep.end_raw,
        timep.interval_minutes,
        use_cache_only=True,
    )
    idx = df.index
    brent = yahoo_csv(paths.data_dir, "Brent_yahoo_1h_2023-01-01_2025-04-01 (1).csv")
    usdr = yahoo_csv(paths.data_dir, "USD_RUB_yahoo_1h_2023-01-01_2025-04-01.csv")
    krate = keyrate_series()
    ext = pd.DataFrame(index=idx)
    ext["Brent"] = brent["Close"].reindex(idx).ffill()
    ext["USD"] = usdr["Close"].reindex(idx).ffill()
    ext["KeyRate"] = krate["KeyRate"].reindex(idx).ffill()
    ext.bfill(limit=1, inplace=True)
    joined = df.join(ext, how="left").loc[timep.period_start : timep.period_end - pd.Timedelta(seconds=1)]
    return tk, joined


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Цепной анализ и кластеризация IMOEX без запуска всей модели")
    ap.add_argument("--out-dir", default="outputs", help="Каталог для результатов")
    ap.add_argument("--freq", default="D", help="Частота для цепного анализа (например, D)")
    ap.add_argument("--clusters", type=int, default=5, help="Число кластеров для режима рынка")
    args = ap.parse_args()

    paths = Paths(out_dir=args.out_dir)
    timep = TimeParams()
    os.makedirs(paths.out_dir, exist_ok=True)

    try:
        tk, df = _load_imoex(paths, timep)
    except FileNotFoundError as exc:
        print("Не найдены исходные данные. Сначала выполните основную пайплайн-команду cli.py.")
        raise SystemExit(exc)

    chain_growth_report(df["Close"], tk, os.path.join(paths.out_dir, "imoex_analysis"), freq=args.freq)
    cluster_regimes_for_ticker(tk, df, os.path.join(paths.out_dir, "clustering"), k=args.clusters)
    print("Аналитические таблицы сохранены в outputs/imoex_analysis и outputs/clustering.")


if __name__ == "__main__":
    main()
