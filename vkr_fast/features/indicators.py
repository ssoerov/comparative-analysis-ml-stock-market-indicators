import numpy as np
import pandas as pd
import ta


def add_indicators(df: pd.DataFrame, n_jobs: int = 4) -> pd.DataFrame:
    """Compute common technical indicators (МОЕХ-friendly) and fill gaps.

    Adds: SMA_{5,10,20}, EMA_{5,10,20}, Bollinger bands (BBH, BBL),
    RSI_50, Stochastic (StochK, StochD), ATR_50, OBV,
    MACD (и его сигнал/расхождение), ADX, CCI, ROC, Williams %R.
    """
    out = df.copy()
    close = out["Close"]

    sma_windows = (5, 10, 20)
    ema_windows = (5, 10, 20)

    for w in sma_windows:
        out[f"SMA_{w}"] = ta.trend.sma_indicator(close, w).values
    for w in ema_windows:
        out[f"EMA_{w}"] = ta.trend.ema_indicator(close, w).values

    bb = ta.volatility.BollingerBands(close, 20)
    out["BBH"], out["BBL"] = bb.bollinger_hband().values, bb.bollinger_lband().values

    out["RSI_50"] = ta.momentum.rsi(close, window=50)
    sto = ta.momentum.StochasticOscillator(out["High"], out["Low"], close)
    out["StochK"], out["StochD"] = sto.stoch(), sto.stoch_signal()
    out["ATR_50"] = ta.volatility.average_true_range(out["High"], out["Low"], close, window=50)
    out["OBV"] = ta.volume.on_balance_volume(close, out["Volume"])
    macd = ta.trend.MACD(close)
    out["MACD"] = macd.macd()
    out["MACD_SIGNAL"] = macd.macd_signal()
    out["MACD_DIFF"] = macd.macd_diff()
    out["ADX_14"] = ta.trend.adx(out["High"], out["Low"], out["Close"])
    out["CCI_20"] = ta.trend.cci(out["High"], out["Low"], out["Close"])
    out["ROC_10"] = ta.momentum.roc(out["Close"], window=10)
    out["WILLR_14"] = ta.momentum.williams_r(out["High"], out["Low"], out["Close"])

    # Важно: только прямое заполнение пропусков, без bfill чтобы исключить утечку будущего
    out.ffill(inplace=True)
    # Оставшиеся NaN (из-за начальных окон) лучше удалить позже на этапе make_lags
    return out
