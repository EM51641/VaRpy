from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BaseVar(ABC):
    def __init__(self, theta: float, horizon: int):
        """
        Initialize the BaseVar class.

        Args:
            ret: NDArray[np.float64] - The return series.
            theta: float - The value at risk level (e.g. 0.01 for 99% VaR)
            horizon: int - The forecast horizon.
        """
        if not 0 <= theta <= 1:
            raise ValueError("theta must be between 0 and 1")
        if horizon <= 0:
            raise ValueError("horizon must be greater than 0")

        self.theta = theta
        self.horizon = horizon
        self._var: float | None = None
        self._cvar: float | None = None

    @property
    def var(self) -> float | None:
        return self._var

    @property
    def cvar(self) -> float | None:
        return self._cvar

    @abstractmethod
    def run(self, ret: NDArray[np.float64]) -> None:
        pass
