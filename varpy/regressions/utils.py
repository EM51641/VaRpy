from typing import Tuple

import numpy as np
from numpy.typing import NDArray


class InnovationProcessor:
    """
    A class for processing financial return innovations and extracting excess values.

    This class handles the calculation and processing of innovations (standardized returns)
    and provides methods to extract excess innovations based on a retention threshold.

    Attributes:
        returns (NDArray[np.float64]): The input return series
        mean (float): The mean of the return series
        volatility (float): The volatility of the return series
        retention_threshold (float): The proportion of data to retain (0 to 1)
        retention_index (int): The number of observations to retain
    """

    def __init__(
        self,
        returns: NDArray[np.float64],
        mean: float,
        volatility: float,
        retention_threshold: float,
    ) -> None:
        """
        Initialize the InnovationProcessor.

        Args:
            returns: Array of financial returns
            mean: Mean of the return series
            volatility: Volatility of the return series
            retention_threshold: Proportion of data to retain (0 to 1)
        """
        if not 0 <= retention_threshold <= 1:
            raise ValueError("retention_threshold must be between 0 and 1")

        self.returns = returns
        self.mean = mean
        self.volatility = volatility
        self.retention_threshold = retention_threshold
        self.retention_index = int(
            np.round(self.retention_threshold * len(self.returns))
        )

    def extract_excess_innovation_evt(
        self,
    ) -> Tuple[NDArray[np.float64], float]:
        """
        Extract excess innovations and returns for Extreme Value Theory (EVT) analysis.

        Returns:
            Tuple containing:
            - excess_innovations: Array of excess innovations
            - excess_returns: Array of excess returns
            - threshold: The innovation threshold value
        """
        sorted_innovations = self._generate_sorted_innovations()

        # Get the threshold values
        threshold_innovations = sorted_innovations[: self.retention_index]

        # Calculate excess values
        threshold_value = threshold_innovations[-1]
        excess_innovations = threshold_innovations - threshold_value

        return excess_innovations, threshold_value

    def extract_excess_innovation(self) -> NDArray[np.float64]:
        """
        Extract all sorted innovations.

        Returns:
            Array of sorted innovations (negative values)
        """
        return self._generate_sorted_innovations()

    def _generate_sorted_innovations(self) -> NDArray[np.float64]:
        """
        Generate sorted innovations (standardized returns).

        Returns:
            Array of sorted innovations in descending order
        """
        innovations = (self.returns - self.mean) / self.volatility
        innovations = innovations[~np.isnan(innovations)]
        return -np.sort(innovations)
