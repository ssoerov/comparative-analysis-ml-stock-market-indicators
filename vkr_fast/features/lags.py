import pandas as pd
import numpy as np


# Экзогенные факторы, которые действительно есть на часовом таймфрейме
EXOG_COLUMNS = ("Brent", "USD")


def make_lags(df: pd.DataFrame, window: int, exog_lags: int = 24, target: str = "dclose") -> pd.DataFrame:
    """Create supervised dataset with price and lagged exogenous features only.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe that already содержит индикаторы и уровни факторов.
    window : int
        Number of price lags (lag1..lag_window).
    exog_lags : int
        Number of historical lags per exogenous factor (Brent, USD, KeyRate).
    """
    out = df.copy()
    target = target.lower()
    if target == "logret":
        # Логарифмическая доходность t = Close_t / Close_{t-1}
        out["y"] = np.log(out["Close"] / out["Close"].shift(1))
    else:
        out["y"] = out["Close"].diff().shift(-1)

    # price lag embeddings
    timestamps = out.index[window:]
    features = [out.iloc[i - window : i]["Close"].values for i in range(window, len(out))]
    lag_df = pd.DataFrame(features, columns=[f"lag{l}" for l in range(1, window + 1)])
    lag_df.index = timestamps
    out = out.join(lag_df, how="inner")

    # lag only exogenous columns (strictly t-1..t-L)
    if exog_lags > 0:
        for col in EXOG_COLUMNS:
            if col not in out.columns:
                continue
            for lag in range(1, exog_lags + 1):
                out[f"{col}_lag{lag}"] = out[col].shift(lag)
            out.drop(columns=[col], inplace=True)

    out = out.dropna().reset_index()
    # unify datetime column name
    if "Datetime" not in out.columns:
        if "index" in out.columns:
            out.rename(columns={"index": "Datetime"}, inplace=True)
        elif out.columns[0] not in ("Close", "Open", "High", "Low", "Volume", "y"):
            out.rename(columns={out.columns[0]: "Datetime"}, inplace=True)
        elif "begin" in out.columns:
            out.rename(columns={"begin": "Datetime"}, inplace=True)
    return out
