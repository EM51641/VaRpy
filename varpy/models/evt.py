import numpy as np
import scipy.stats
from numpy.typing import NDArray

from varpy.models.base import BaseVar
from varpy.regressions.arch import garch_forecast, garch_regression
from varpy.regressions.utils import InnovationProcessor


class EVT(BaseVar):
    def __init__(self, theta: float, horizon: int):
        super().__init__(theta, horizon)

    def run(self, ret: NDArray[np.float64]) -> None:
        """
        Compute VaR and CVaR using Extreme Value Theory.

        Returns:
            None
        """
        # Get GARCH forecasts
        mean, var, cond_vol = self._get_garch_forecasts(ret)

        # Get excess innovations and fit GPD
        excess_innovations, last_innovation = self._get_excess_innovations(
            ret, mean, cond_vol
        )
        innovation_params = self._fit_gpd(excess_innovations)

        # Compute unconditional VaR
        uncond_var = self._compute_unconditional_var(
            innovation_params, last_innovation, excess_innovations.size, ret.size
        )

        # Compute final VaR and CVaR
        self._var = self._compute_var(mean, var, uncond_var)
        self._cvar = self._compute_cvar(
            mean, var, uncond_var, innovation_params, last_innovation
        )

    def _get_garch_forecasts(
        self, ret: NDArray[np.float64]
    ) -> tuple[float, float, float]:
        """
        Get GARCH forecasts for mean, variance, and conditional volatility.

        Returns:
            Tuple of (mean, variance, conditional_volatility)
        """
        reg = garch_regression(ret)
        return garch_forecast(reg, self.horizon)

    def _get_excess_innovations(
        self, ret: NDArray[np.float64], mean: float, cond_vol: float
    ) -> tuple[NDArray[np.float64], float]:
        """
        Extract excess innovations using the innovation processor.

        Args:
            mean: Forecasted mean
            cond_vol: Conditional volatility

        Returns:
            Tuple of (excess_innovations, last_innovation)
        """
        ip = InnovationProcessor(ret, mean, cond_vol, 0.10)
        return ip.extract_excess_innovation_evt()

    def _fit_gpd(self, excess_innovations: NDArray[np.float64]) -> tuple[float, ...]:
        """
        Fit Generalized Pareto Distribution to excess innovations.

        Args:
            excess_innovations: Array of excess innovations

        Returns:
            Tuple of (shape, location, scale) parameters

        Raises:
            ValueError: If GPD parameters are invalid
        """
        params = scipy.stats.genpareto.fit(excess_innovations, floc=0)

        if params[0] + params[2] < 0:
            raise ValueError(
                "Invalid GPD parameters. Try using more data for better estimation."
            )

        return params

    def _compute_unconditional_var(
        self,
        innovation_params: tuple[float, ...],
        last_innovation: float,
        excess_size: int,
        ret_size: int,
    ) -> float:
        """
        Compute unconditional VaR using EVT parameters.

        Args:
            innovation_params: GPD parameters (shape, location, scale)
            last_innovation: Last innovation value
            excess_size: Number of excess observations

        Returns:
            Unconditional VaR value
        """
        shape, _, scale = innovation_params
        return last_innovation + (scale / shape) * (
            (ret_size * self.theta / excess_size) ** (-shape) - 1
        )

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

    def _compute_cvar(
        self,
        mean: float,
        var: float,
        uncond_var: float,
        innovation_params: tuple[float, ...],
        last_innovation: float,
    ) -> float:
        """
        Compute final CVaR value.

        Args:
            mean: Forecasted mean
            var: Forecasted variance
            uncond_var: Unconditional VaR
            innovation_params: GPD parameters (shape, location, scale)
            last_innovation: Last innovation value

        Returns:
            CVaR value
        """
        shape, _, scale = innovation_params
        return -(
            mean
            + uncond_var
            * np.sqrt(var)
            * (
                1 / (1 - shape)
                + (scale - shape * last_innovation) / ((1 - shape) * uncond_var)
            )
        )
