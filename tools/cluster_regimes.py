import os
import sys
from typing import Dict, List

import pandas as pd

# ensure local package import
sys.path.append(os.getcwd())
from vkr_fast.config import Paths, TimeParams, TICKERS
from vkr_fast.data import fetch_moex, yahoo_csv, keyrate_series
from vkr_fast.analysis import cluster_regimes_for_ticker


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _load_raw_cached(paths: Paths, timep: TimeParams, tickers: Dict[str, tuple]) -> Dict[str, pd.DataFrame]:
    """Load aligned MOEX+macro using only cache/CSV (no network)."""
    brent = yahoo_csv(paths.data_dir, "Brent_yahoo_1h_2023-01-01_2025-04-01 (1).csv")
    usdr = yahoo_csv(paths.data_dir, "USD_RUB_yahoo_1h_2023-01-01_2025-04-01.csv")
    krate = keyrate_series()
    out = {}
    for tk in tickers:
        df = fetch_moex(
            paths.cache_dir,
            tk,
            *tickers[tk],
            timep.start_raw,
            timep.end_raw,
            timep.interval_minutes,
            use_cache_only=True,
        )
        idx = df.index
        ext = pd.DataFrame(index=idx)
        ext["Brent"] = brent["Close"].reindex(idx).ffill()
        ext["USD"] = usdr["Close"].reindex(idx).ffill()
        ext["KeyRate"] = krate["KeyRate"].reindex(idx).ffill()
        ext.bfill(limit=1, inplace=True)
        out[tk] = df.join(ext, how="left").loc[timep.period_start : timep.period_end - pd.Timedelta(seconds=1)]
    return out


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Кластеризация режимов рынка по каждому тикеру (IMOEX-only)")
    ap.add_argument("--k", type=int, default=5, help="Число кластеров KMeans")
    ap.add_argument("--win-vol", type=int, default=50, help="Окно для расчёта волатильности")
    ap.add_argument("--out-dir", default="outputs/clustering", help="Каталог для результатов")
    args = ap.parse_args()

    paths = Paths()
    timep = TimeParams()
    tickers = TICKERS
    _ensure_dir(args.out_dir)

    # Загружаем только из кеша/локальных CSV
    try:
        raw = _load_raw_cached(paths, timep, tickers)
    except FileNotFoundError as exc:
        print("Не найден кеш MOEX. Сначала выполните онлайн‑запуск cli.py, чтобы заполнить data_cache/. Ошибка:", exc)
        return

    any_done = False
    for tk, df in raw.items():
        res = cluster_regimes_for_ticker(tk, df, args.out_dir, k=args.k, win_vol=args.win_vol)
        if res is None:
            print(f"[{tk}] Недостаточно данных для кластеризации.")
        else:
            any_done = True
            print(f"[{tk}] Кластеры сохранены в {args.out_dir}")

    if any_done:
        print("Кластеризация режимов завершена: outputs/clustering/")
    else:
        print("Кластеризация не выполнена: нет достаточных данных.")


if __name__ == "__main__":
    main()
