import math
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def dm_test(
    e1: np.ndarray,
    e2: np.ndarray,
    h: int = 1,
    loss: str = "mse",
    input_is_loss: bool = False,
):
    """Diebold–Mariano test on comparable loss series or precomputed losses.

    Parameters
    ----------
    e1, e2 : arrays of forecast errors (same length or will be aligned), or loss arrays if input_is_loss=True
    h : forecast horizon (used for HAC bandwidth)
    loss : currently supports "mse" (squared error) and "mae" (absolute error) when input_is_loss=False
    input_is_loss : if True, treat e1/e2 as already computed loss series (no transformation)
    """
    if input_is_loss:
        l1 = np.asarray(e1, dtype=float).ravel()
        l2 = np.asarray(e2, dtype=float).ravel()
    else:
        e1 = np.asarray(e1, dtype=float).ravel()
        e2 = np.asarray(e2, dtype=float).ravel()
        L = min(len(e1), len(e2))
        if L == 0:
            return float("nan"), float("nan")
        e1 = e1[:L]
        e2 = e2[:L]

        if loss.lower() == "mae":
            l1 = np.abs(e1)
            l2 = np.abs(e2)
        else:
            # Default to squared-error loss (unbiased DM for regression forecasts)
            l1 = np.square(e1)
            l2 = np.square(e2)

    d = l1 - l2
    n = len(d)
    m = d.mean()
    if not np.isfinite(m):
        return float("nan"), float("nan")

    # For horizon h the DM variance uses truncation at h-1; fall back to var when h<=1
    bandwidth = max(h - 1, 0)
    if bandwidth == 0:
        nw = np.var(d, ddof=1) if n > 1 else 0.0
    else:
        nw = _newey_west_lrv(d, bandwidth=bandwidth)

    if nw <= 0:
        return float("nan"), float("nan")

    stat = np.sqrt(n) * m / np.sqrt(nw)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return stat, p


def block_bootstrap_means(e: np.ndarray, B: int = 600, block_len: Optional[int] = None) -> np.ndarray:
    """Moving block bootstrap for the mean of errors to respect dependence.

    - e: 1D array of errors (time-ordered)
    - B: number of bootstrap replications
    - block_len: block length; default sqrt(n) (rounded)
    Returns array of bootstrapped means length B.
    """
    e = np.asarray(e)
    n = len(e)
    if n == 0:
        return np.array([])
    if block_len is None:
        block_len = max(2, int(np.sqrt(n)))
    block_len = max(1, min(block_len, n))
    # number of blocks to cover length n
    k = max(1, int(np.ceil(n / block_len)))
    out = np.empty(B, dtype=float)
    for b in range(B):
        # sample starting indices for blocks with replacement
        max_start = max(1, n - block_len + 1)
        starts = np.random.randint(0, max_start, size=k)
        sample = np.concatenate([e[s:s + block_len] for s in starts])[:n]
        out[b] = sample.mean()
    return out


def _newey_west_lrv(x: np.ndarray, bandwidth: Optional[int] = None) -> float:
    """Newey–West long-run variance estimator with Bartlett kernel."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float(np.var(x, ddof=1) if n > 1 else 0.0)
    if bandwidth is None:
        # Common rule-of-thumb bandwidth
        bandwidth = max(1, int(4 * (n / 100) ** (2 / 9)))
    x = x - x.mean()
    gamma0 = np.dot(x, x) / n
    lrv = gamma0
    for k in range(1, min(bandwidth, n - 1) + 1):
        w = 1 - k / (bandwidth + 1)
        cov = np.dot(x[:-k], x[k:]) / n
        lrv += 2 * w * cov
    return float(max(lrv, 1e-18))


def bars_per_day_from_datetime(dt: pd.Series) -> float:
    """Estimate average number of bars per trading day from a Datetime series."""
    dt = pd.to_datetime(dt)
    by_day = dt.dt.tz_convert("UTC") if getattr(dt.dt, 'tz', None) is not None else dt
    counts = by_day.dt.date.value_counts()
    if len(counts) == 0:
        return 24.0
    # median is robust to partial days
    return float(np.median(counts.values))


def sharpe_annualized(returns: np.ndarray,
                      dt_index: pd.Series,
                      use_hac: bool = True,
                      trading_days_per_year: float = 252.0) -> float:
    """Annualized Sharpe with correct sampling frequency and optional HAC variance.

    returns: per-bar strategy returns (including zeros when flat)
    dt_index: aligned timestamps for returns (len == len(returns))
    use_hac: if True, use Newey–West LRV in denominator
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n == 0:
        return float("nan")
    mu = r.mean()
    if use_hac:
        var_bar = _newey_west_lrv(r)
    else:
        var_bar = np.var(r, ddof=1) if n > 1 else 0.0
    if var_bar <= 0:
        return float("nan")
    bpd = bars_per_day_from_datetime(pd.Series(dt_index))
    samp_per_year = trading_days_per_year * bpd
    return float(mu / np.sqrt(var_bar) * np.sqrt(samp_per_year))


def _block_bootstrap_series(x: np.ndarray, B: int = 600, block_len: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.empty((0,))
    if block_len is None:
        block_len = max(2, int(np.sqrt(n)))
    block_len = max(1, min(block_len, n))
    k = max(1, int(np.ceil(n / block_len)))
    out = np.empty((B, n), dtype=float)
    for b in range(B):
        max_start = max(1, n - block_len + 1)
        starts = np.random.randint(0, max_start, size=k)
        sample = np.concatenate([x[s:s + block_len] for s in starts])[:n]
        out[b] = sample
    return out


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    dt_index: pd.Series,
    B: int = 600,
    block_len: Optional[int] = None,
    alpha: float = 0.05,
    use_hac: bool = True,
) -> Tuple[float, float, float]:
    """Moving block bootstrap CI for annualized Sharpe; returns (low, med, high)."""
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return (float("nan"), float("nan"), float("nan"))
    samples = _block_bootstrap_series(r, B=B, block_len=block_len)
    dt_series = pd.Series(dt_index)
    bpd = bars_per_day_from_datetime(dt_series)
    samp_per_year = 252.0 * bpd
    vals = []
    for i in range(samples.shape[0]):
        rr = samples[i]
        if use_hac:
            var_bar = _newey_west_lrv(rr)
        else:
            var_bar = np.var(rr, ddof=1) if len(rr) > 1 else 0.0
        if var_bar <= 0:
            vals.append(np.nan)
        else:
            vals.append(rr.mean() / np.sqrt(var_bar) * np.sqrt(samp_per_year))
    arr = np.array(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    lo = float(np.percentile(arr, 100 * (alpha / 2)))
    med = float(np.percentile(arr, 50))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return (lo, med, hi)
