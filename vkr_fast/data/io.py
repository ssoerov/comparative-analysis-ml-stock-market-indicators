import os
from io import StringIO

import numpy as np
import pandas as pd


def yahoo_csv(data_dir: str, fname: str) -> pd.DataFrame:
    path = os.path.join(data_dir, fname)
    df = pd.read_csv(path)
    if "Datetime" not in df.columns:
        df.columns = ["Datetime", "Close", "High", "Low", "Open", "Volume"]
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce", utc=True)
    df.set_index("Datetime", inplace=True)
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().astype("float32")
    return df


def keyrate_series() -> pd.DataFrame:
    txt = """Дата Ставка
28.10.2024 21,00
16.09.2024 19,00
29.07.2024 18,00
18.12.2023 16,00
30.10.2023 15,00
18.09.2023 13,00
15.08.2023 12,00
24.07.2023 8,50"""
    df = pd.read_csv(
        StringIO(txt.replace(",", ".")),
        sep=r"\s+",
        names=["date", "rate"],
        parse_dates=["date"],
        dayfirst=True,
        header=None,
        skiprows=1,
    )
    df.set_index("date", inplace=True)
    df.index = df.index.tz_localize("UTC")
    hrs = pd.date_range("2023-01-01", "2025-04-01", freq="h", tz="UTC")
    out = pd.DataFrame(index=hrs, data={"KeyRate": df["rate"]})
    out.ffill(inplace=True)
    return out.astype("float32")

