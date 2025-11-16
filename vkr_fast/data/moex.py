import os
from datetime import timedelta
from typing import Dict

import pandas as pd
import requests


def fetch_moex(
    cache_dir: str,
    tk: str,
    market: str,
    board: str,
    secid: str,
    start_raw: pd.Timestamp,
    end_raw: pd.Timestamp,
    interval_minutes: int,
    use_cache_only: bool = False,
) -> pd.DataFrame:
    cache = os.path.join(
        cache_dir,
        f"{tk}_{interval_minutes}m_{start_raw.date()}_{end_raw.date()}.csv",
    )
    if os.path.exists(cache):
        # Assume timestamps in cache are already UTC; try tz_convert
        try:
            return pd.read_csv(cache, index_col=0, parse_dates=True).tz_convert("UTC")
        except Exception:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            df.index = df.index.tz_localize("UTC")
            return df

    if use_cache_only:
        raise FileNotFoundError(
            f"MOEX cache not found for {tk}. Expected at: {cache}. Run online to populate cache."
        )

    url = (
        f"https://iss.moex.com/iss/engines/stock/markets/{market}/boards/"
        f"{board}/securities/{secid}/candles.json"
    )

    rows, cur, step = [], start_raw, timedelta(days=25)
    while cur < end_raw:
        till = min(cur + step, end_raw)
        params = {
            "from": cur.date(),
            "till": till.date(),
            "interval": interval_minutes,
            "iss.meta": "off",
            "iss.only": "candles",
            "iss.zone": "UTC",
        }
        try:
            js = requests.get(url, params=params, timeout=20).json()
        except Exception as exc:
            # If network fails, try to fall back to cache if it appeared during partial fetch
            if os.path.exists(cache):
                try:
                    return pd.read_csv(cache, index_col=0, parse_dates=True).tz_convert("UTC")
                except Exception:
                    dfc = pd.read_csv(cache, index_col=0, parse_dates=True)
                    dfc.index = dfc.index.tz_localize("UTC")
                    return dfc
            raise RuntimeError(f"Failed to fetch MOEX data for {tk}: {exc}")
        rows += js["candles"]["data"]
        cols = js["candles"]["columns"]
        cur = till + timedelta(days=1)

    df = pd.DataFrame(rows, columns=cols)
    df["begin"] = pd.to_datetime(df["begin"], utc=True)
    df.set_index("begin", inplace=True)
    keep = ["open", "high", "low", "close", "value", "volume"]
    df = df[keep]
    df.columns = map(str.title, df.columns)
    df.to_csv(cache)
    return df
