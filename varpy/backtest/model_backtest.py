import numpy as np
from numpy.typing import NDArray

from varpy.models.base import BaseVar

DEFAULT_STEP = 1
DEFAULT_ROLLING_WINDOW = 500


class Backtest:
    """
    Backtest the VaR model and test its time significance

    Parameters
    -------
    ret: np.ndarray
    step: np.int64
    model: BaseVar

    Returns
    -------
    var: np.ndarray
    cvar: np.ndarray
    """

    def __init__(
        self,
        ret: NDArray[np.float64],
        model: BaseVar,
        step: int = DEFAULT_STEP,
        rolling_window: int = DEFAULT_ROLLING_WINDOW,
    ):
        """
        Initialize the Backtest class
        """
        self.ret = ret
        self.step = step
        self.model = model
        self.rolling_window = rolling_window

        self.var: NDArray[np.float64] | None = None
        self.cvar: NDArray[np.float64] | None = None

    def simulation(self) -> None:
        """
        Simulate the VaR and CVaR for the given model
        """
        self.var, self.cvar = np.ones(self.ret.size), np.ones(self.ret.size)

        for i in range(self.rolling_window, self.ret.size, self.step):
            ret = self.ret[i - self.rolling_window : i]
            try:
                self.model.run(ret)
                self.var[i : i + self.step], self.cvar[i : i + self.step] = (
                    self.model.var,
                    self.model.cvar,
                )
            except Exception:
                self.var[i : i + self.step], self.cvar[i : i + self.step] = (
                    np.nan,
                    np.nan,
                )
