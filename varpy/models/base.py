from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from varpy.regressions.utils import InnovationProcessor

DEFAULT_RETENTION_THRESHOLD = 0.10


class BaseVar(ABC):
    def __init__(
        self,
        theta: float,
        horizon: int,
        retention_threshold: float = DEFAULT_RETENTION_THRESHOLD,
    ):
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
        self.retention_threshold = retention_threshold
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

    def _get_innovation_processor(
        self, ret: NDArray[np.float64], mean: float, cond_vol: NDArray[np.float64]
    ) -> InnovationProcessor:
        """
        Get the innovation processor.

        Args:
            mean: Forecasted mean
            cond_vol: Conditional volatility

        Returns:
            InnovationProcessor
        """
        return InnovationProcessor(ret, mean, cond_vol, self.retention_threshold)
