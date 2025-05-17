import numpy as np
import scipy.stats
from numpy.typing import NDArray

from varpy.models.base import BaseVar
from varpy.regressions.arch import garch_forecast, garch_regression
from varpy.regressions.utils import InnovationProcessor


class Normal(BaseVar):
    def __init__(self, theta: float, horizon: int):
        super().__init__(theta, horizon)

    def run(self, ret: NDArray[np.float64]) -> None:
        """
        Compute VaR and CVaR using Normal distribution assumption.

        Args:
            ret: Array of returns

        Returns:
            None
        """
        # Get GARCH forecasts
        mean, var, cond_vol = self._get_garch_forecasts(ret)

        # Get excess innovations and fit normal distribution
        excess_innovations = self._get_excess_innovations(ret, mean, cond_vol)
        mu, scale = self._fit_normal(excess_innovations)

        # Compute unconditional VaR
        uncond_var = self._compute_unconditional_var(scale, mu)

        # Compute final VaR and CVaR
        self._var = self._compute_var(mean, var, uncond_var)
        self._cvar = self._compute_cvar(mean, var)

    def _get_garch_forecasts(
        self, ret: NDArray[np.float64]
    ) -> tuple[float, float, float]:
        """
        Get GARCH forecasts for mean, variance, and conditional volatility.

        Args:
            ret: Array of returns

        Returns:
            Tuple of (mean, variance, conditional_volatility)
        """
        reg = garch_regression(ret, "gaussian")
        return garch_forecast(reg, self.horizon)

    def _get_excess_innovations(
        self, ret: NDArray[np.float64], mean: float, cond_vol: float
    ) -> NDArray[np.float64]:
        """
        Extract excess innovations using the innovation processor.

        Args:
            ret: Array of returns
            mean: Forecasted mean
            cond_vol: Conditional volatility

        Returns:
            Array of excess innovations
        """
        ip = InnovationProcessor(ret, mean, cond_vol, 0.10)
        return ip.extract_excess_innovation()

    def _fit_normal(
        self, excess_innovations: NDArray[np.float64]
    ) -> tuple[float, float]:
        """
        Fit Normal distribution to excess innovations.

        Args:
            excess_innovations: Array of excess innovations

        Returns:
            Tuple of (mean, scale) parameters
        """
        return scipy.stats.norm.fit(excess_innovations)

    def _compute_unconditional_var(self, scale: float, mu: float) -> float:
        """
        Compute unconditional VaR using Normal distribution parameters.

        Args:
            scale: Scale parameter of normal distribution
            mu: Mean parameter of normal distribution

        Returns:
            Unconditional VaR value
        """
        return scipy.stats.norm.ppf(1 - self.theta) * scale - mu

    def _compute_var(self, mean: float, var: float, uncond_var: float) -> float:
        """
        Compute final VaR value.

        Args:
            mean: Forecasted mean
            var: Forecasted variance
            uncond_var: Unconditional VaR

        Returns:
            VaR value
        """
        return -(mean + np.sqrt(var) * uncond_var)

    def _compute_cvar(self, mean: float, var: float) -> float:
        """
        Compute final CVaR value.

        Args:
            mean: Forecasted mean
            var: Forecasted variance

        Returns:
            CVaR value
        """
        return -(
            mean
            + np.sqrt(var)
            * (scipy.stats.norm.pdf(scipy.stats.norm.ppf(1 - self.theta)) / self.theta)
        )
