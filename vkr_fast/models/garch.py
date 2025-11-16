from typing import Tuple, Optional

import numpy as np
from arch import arch_model


def fit_garch(
    residuals: np.ndarray,
    vol: str = "GJR-GARCH",
    p: int = 1,
    q: int = 1,
    dist: str = "t",
):
    """Fit a volatility model to residuals (mean already handled upstream).

    vol: one of {"GARCH", "GJR-GARCH", "EGARCH"}
    dist: one of {"normal", "t"}
    Returns fitted result (arch.univariate.base.ARCHModelResult)
    """
    res = np.asarray(residuals, dtype=float)
    vol = vol.upper()
    if vol not in {"GARCH", "GJR-GARCH", "EGARCH"}:
        vol = "GJR-GARCH"
    if dist not in {"normal", "t"}:
        dist = "t"
    am = arch_model(res, mean="Zero", vol=vol, p=p, q=q, dist=dist)
    fitted = am.fit(disp="off")
    return fitted


def forecast_garch(
    fitted,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Forecast conditional mean and sigma for given horizon.

    Since mean='Zero', mu forecast is zeros. Returns (mu_fc, sigma_fc).
    """
    f = fitted.forecast(horizon=horizon, reindex=False)
    mu_fc = np.zeros(horizon, dtype=float)
    # variance forecasts at steps 1..horizon
    var_fc = f.variance.values[-1]
    if var_fc.shape[0] != horizon:
        var_fc = np.resize(var_fc, horizon)
    sigma_fc = np.sqrt(np.maximum(var_fc, 0.0))
    return mu_fc, sigma_fc

