import numpy as np
import scipy.stats
from numpy.typing import NDArray

from varpy.models.base import DEFAULT_RETENTION_THRESHOLD, BaseVar
from varpy.regressions.arch import garch_forecast, garch_regression


class Student(BaseVar):
    def __init__(
        self,
        theta: float,
        horizon: int,
        retention_threshold: float = DEFAULT_RETENTION_THRESHOLD,
    ):
        super().__init__(theta, horizon, retention_threshold)

    def run(self, ret: NDArray[np.float64]) -> None:
        """
        Compute VaR and CVaR using Student's t-distribution assumption.

        Args:
            ret: Array of returns

        Returns:
            None
        """
        # Get GARCH forecasts
        mean, var, cond_vol = self._get_garch_forecasts(ret)

        # Get excess innovations and fit t-distribution
        excess_innovations = self._get_excess_innovations(ret, mean, cond_vol)
        degrees_of_freedom = self._fit_student_t(excess_innovations)

        # Compute unconditional VaR
        uncond_var = self._compute_unconditional_var(degrees_of_freedom)

        # Compute final VaR and CVaR
        self._var = self._compute_var(mean, var, uncond_var)
        self._cvar = self._compute_cvar(mean, var, degrees_of_freedom)

    def _get_garch_forecasts(
        self, ret: NDArray[np.float64]
    ) -> tuple[float, float, NDArray[np.float64]]:
        """
        Get GARCH forecasts for mean, variance, and conditional volatility.

        Args:
            ret: Array of returns

        Returns:
            Tuple of (mean, variance, conditional_volatility)
        """
        reg = garch_regression(ret, "studentst")
        return garch_forecast(reg, self.horizon)

    def _get_excess_innovations(
        self, ret: NDArray[np.float64], mean: float, cond_vol: NDArray[np.float64]
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
        ip = self._get_innovation_processor(ret, mean, cond_vol)
        return ip.extract_excess_innovation()

    def _fit_student_t(self, excess_innovations: NDArray[np.float64]) -> float:
        """
        Fit Student's t-distribution to excess innovations.

        Args:
            excess_innovations: Array of excess innovations

        Returns:
            Degrees of freedom parameter
        """
        return scipy.stats.t.fit(excess_innovations)[0]

    def _compute_unconditional_var(self, degrees_of_freedom: float) -> float:
        """
        Compute unconditional VaR using Student's t-distribution parameters.

        Args:
            degrees_of_freedom: Degrees of freedom parameter

        Returns:
            Unconditional VaR value
        """
        return np.sqrt(
            (degrees_of_freedom - 2) / degrees_of_freedom
        ) * scipy.stats.t.ppf(1 - self.theta, degrees_of_freedom)

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
        self, mean: float, var: float, degrees_of_freedom: float
    ) -> float:
        """
        Compute final CVaR value.

        Args:
            mean: Forecasted mean
            var: Forecasted variance
            degrees_of_freedom: Degrees of freedom parameter

        Returns:
            CVaR value
        """
        t_quantile = scipy.stats.t.ppf(1 - self.theta, degrees_of_freedom)
        return -(
            mean
            + (scipy.stats.t.pdf(t_quantile, degrees_of_freedom) / self.theta)
            * (degrees_of_freedom + t_quantile**2)
            / (degrees_of_freedom - 1)
            * np.sqrt(var)
        )
