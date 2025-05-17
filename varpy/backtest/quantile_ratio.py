import numpy as np
from numpy.typing import NDArray


def quantile_ratio(var: NDArray[np.float64], cvar: NDArray[np.float64]) -> float:
    """
    Calculate the quantile ratio of the VaR and CVaR.

    Args:
        var: The VaR values.
        cvar: The CVaR values.

    Returns:
        The mean quantile ratio of the VaR and CVaR.
    """
    return float(np.mean(cvar / var))
