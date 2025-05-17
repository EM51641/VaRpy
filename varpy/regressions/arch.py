from typing import Literal

import numpy as np
from arch import arch_model
from arch.univariate.base import ARCHModelResult

DIST = Literal[
    "normal",
    "gaussian",
    "t",
    "studentst",
    "skewstudent",
    "skewt",
    "ged",
    "generalized error",
]


def garch_regression(
    ret: np.typing.NDArray[np.float64],
    dist: DIST = "gaussian",
) -> ARCHModelResult:
    """
    Perform ARCH regression on the given return series.

    Parameters:
    ret (np.ndarray): The return series to perform the regression on.
    dist (str): The distribution to use for the model.

    Returns:
    forecasted_mean (float): The forecasted mean of the return series.
    forecasted_var (float): The forecasted variance of the return series.
    conditional_volatility (float): The conditional volatility of the return series.
    """
    am = arch_model(ret, vol="GARCH", p=1, o=1, q=1, mean="AR", lags=1, dist=dist)
    res = am.fit(update_freq=1, disp="off")
    return res


def garch_forecast(
    res: ARCHModelResult, horizon: int = 1
) -> tuple[float, float, np.typing.NDArray[np.float64]]:
    """
    Forecast the mean, variance, and conditional volatility of the return series.

    Parameters:
    res (ARCHModelResult): The result of the ARCH regression.
    horizon (int): The horizon of the forecast.

    Returns:
    mean_forecast (float): The forecasted mean of the return series.
    var_forecast (float): The forecasted variance of the return series.
    cond_vol (np.ndarray[np.float64]): The conditional volatility of the return series.
    """
    forecasts = res.forecast(horizon=horizon)
    mean_forecast: float = forecasts.mean.dropna().iloc[-1, -1]  # type: ignore
    var_forecast: float = forecasts.variance.dropna().iloc[-1, -1]  # type: ignore
    cond_vol = np.asarray(res.conditional_volatility, dtype=np.float64)
    return mean_forecast, var_forecast, cond_vol
