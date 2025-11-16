import numpy as np
import statsmodels.api as sm


def fit_sarimax(y_train: np.ndarray, X_train: np.ndarray):
    model = sm.tsa.statespace.SARIMAX(
        y_train,
        exog=X_train,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        trend="c",
        simple_differencing=False,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=50)
    return model


def sarimax_predict(model, start: int, end: int, exog_future: np.ndarray) -> np.ndarray:
    return model.predict(start=start, end=end, exog=exog_future)
