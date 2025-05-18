from typing import Optional

import numpy as np
import scipy
from numpy.typing import NDArray

BOOTSTRAP_UPPER_BOUND = 1.962
BOOTSTRAP_LOWER_BOUND = -1.962
BOOTSTRAP_ITERATIONS = 1000
MIN_BOOTSTRAP_SIZE = 256


class BacktestResults:
    """Class to store and compute backtesting results."""

    def __init__(
        self,
        ret: NDArray[np.float64],
        var: NDArray[np.float64],
        cvar: NDArray[np.float64],
        theta: float,
    ):
        """
        Initialize backtesting results.

        Args:
            ret: Array of returns
            var: Value at Risk
            cvar: Conditional Value at Risk
            theta: Expected probability of violation
        """
        self.ret = ret
        self.var = var
        self.cvar = cvar
        self.theta = theta

        # Initialize attributes with proper types
        self.var_violations: Optional[int] = None
        self.cvar_violations: Optional[int] = None
        self.var_violation_mtx: Optional[NDArray[np.int_]] = None
        self.cvar_violation_mtx: Optional[NDArray[np.int_]] = None
        self.christoffersen: Optional[float] = None
        self.binomial: Optional[float] = None
        self.kupiec: Optional[float] = None
        self.tuff: Optional[float] = None
        self.hass: Optional[float] = None
        self.q_ratio: Optional[float] = None
        self.q_ratio_bootstrap: Optional[float] = None

    def run(self) -> None:
        """Run all backtesting computations."""
        # Compute violations
        self.var_violations = self._count_var_violations()
        self.cvar_violations = self._count_cvar_violations()
        self.var_violation_mtx = self._create_var_violation_matrix()
        self.cvar_violation_mtx = self._create_cvar_violation_matrix()

        # Compute test statistics
        self.christoffersen = self._compute_christoffersen_test()
        self.binomial = self._compute_binomial_test()
        self.kupiec = self._compute_coverage_statistic()
        self.tuff = self._compute_tuff_test()
        self.hass = self._compute_hass_test()
        self.q_ratio = self._compute_quantile_ratio()
        self.q_ratio_bootstrap = self._compute_bootstrap_test()

    def _create_var_violation_matrix(self) -> NDArray[np.int_]:
        """Create binary matrix of violations.

        Returns:
            Binary array where 1 indicates a VaR violation (return < -VaR)
        """
        return np.where(self.ret < self.var, 1, 0)

    def _create_cvar_violation_matrix(self) -> NDArray[np.int_]:
        """Create binary matrix of violations.

        Returns:
            Binary array where 1 indicates a CVaR violation (return < -CVaR)
        """
        return np.where(self.ret < self.cvar, 1, 0)

    def _count_var_violations(self) -> int:
        """Count number of violations."""
        return np.sum(self._create_var_violation_matrix())  # type: ignore

    def _count_cvar_violations(self) -> int:
        """Count number of violations."""
        return np.sum(self._create_cvar_violation_matrix())  # type: ignore

    def _count_transitions(self, matrix: NDArray[np.int_]) -> tuple[int, int, int, int]:
        """
        Count the transitions between violations and non-violations in the matrix.

        Args:
            matrix: Binary array where 0 represents a violation and 1 represents no violation

        Returns:
            tuple of (m_00, m_01, m_10, m_11) where:
            - m_00: transitions from no violation to no violation
            - m_01: transitions from no violation to violation
            - m_10: transitions from violation to no violation
            - m_11: transitions from violation to violation
        """
        m_00 = m_01 = m_10 = m_11 = 0
        for i in range(matrix.size - 1):
            if matrix[i] == 0 and matrix[i + 1] == 0:
                m_00 += 1
            elif matrix[i] == 0 and matrix[i + 1] != 0:
                m_01 += 1
            elif matrix[i] != 0 and matrix[i + 1] == 0:
                m_10 += 1
            elif matrix[i] != 0 and matrix[i + 1] != 0:
                m_11 += 1
        return m_00, m_01, m_10, m_11

    def _compute_independence_statistic(
        self, m_00: int, m_01: int, m_10: int, m_11: int
    ) -> float:
        """
        Compute the independence test statistic (CCI) using likelihood ratio.

        The test statistic is calculated as:
        LR_CCI = -2 * log(((1-π)^(n00+n10) * π^(n01+n11)) /
                          ((1-π0)^n00 * π0^n01 * (1-π1)^n10 * π1^n11))

        where:
        - π is the unconditional probability of violation
        - π0 is the probability of violation given no violation in previous period
        - π1 is the probability of violation given violation in previous period
        - n00, n01, n10, n11 are the transition counts

        Args:
            m_00: transitions from no violation to no violation
            m_01: transitions from no violation to violation
            m_10: transitions from violation to no violation
            m_11: transitions from violation to violation

        Returns:
            Independence test statistic
        """
        total = m_00 + m_01 + m_10 + m_11

        # Calculate probabilities
        pi = (m_01 + m_11) / total  # Unconditional probability
        pi0 = (
            m_01 / (m_00 + m_01) if (m_00 + m_01) > 0 else 0
        )  # Conditional probability given no violation
        pi1 = (
            m_11 / (m_10 + m_11) if (m_10 + m_11) > 0 else 0
        )  # Conditional probability given violation

        # Calculate likelihood ratio
        numerator = (1 - pi) ** (m_00 + m_10) * pi ** (m_01 + m_11)
        denominator = (1 - pi0) ** m_00 * pi0**m_01 * (1 - pi1) ** m_10 * pi1**m_11

        return -2 * np.log(numerator / denominator)

    def _compute_coverage_statistic(self) -> float:
        """
        Compute the coverage test statistic (POF - Proportion of Failures) using likelihood ratio.

        The test statistic is calculated as:
        LR_POF = -2 * log(((1-p)^(N-x) * p^x) / ((1-x/N)^(N-x) * (x/N)^x))

        where:
        - p is the expected probability of violation (theta)
        - N is the total number of observations
        - x is the number of violations

        Returns:
            Coverage test statistic
        """
        assert self.var_violation_mtx is not None
        assert self.var_violations is not None

        n = self.var_violation_mtx.size  # Total number of observations
        x = self.var_violations  # Number of violations

        # Calculate likelihood ratio
        numerator = (1 - self.theta) ** (n - x) * self.theta**x
        denominator = (1 - x / n) ** (n - x) * (x / n) ** x

        return -2 * np.log(numerator / denominator)

    def _find_first_failure(self, matrix: NDArray[np.int_]) -> int:
        """
        Find the index of the first violation in the matrix.

        Args:
            matrix: Binary array of violations

        Returns:
            Index of first violation
        """
        return np.where(matrix == 1)[0][0] + 1

    def _compute_tuff_test(self) -> float:
        """
        Compute the TUFF (Time Until First Failure) test statistic using likelihood ratio.

        The test statistic is calculated as:
        LR_TUFF = -2 * log(p * (1-p)^(n-1) / ((1/n) * (1-1/n)^(n-1)))

        where:
        - p is the expected probability of violation (theta)
        - n is the time until first failure

        Returns:
            TUFF test statistic
        """
        assert self.var_violation_mtx is not None

        n = self._find_first_failure(self.var_violation_mtx)  # Time until first failure

        # Calculate likelihood ratio
        numerator = self.theta * (1 - self.theta) ** (n - 1)
        denominator = (1 / n) * (1 - 1 / n) ** (n - 1)

        return -2 * np.log(numerator / denominator)

    def _compute_time_between_failures(self, matrix: NDArray[np.int_]) -> float:
        """
        Compute the Time Between Failures Independence (TBFI) test statistic using likelihood ratio.

        The test statistic is calculated as:
        LR_TBFI = -2 * sum(log(p * (1-p)^(n_i-1) / ((1/n_i) * (1-1/n_i)^(n_i-1))))

        where:
        - p is the expected probability of violation (theta)
        - n_i is the time between failures
        - The sum is over all failure intervals

        Args:
            matrix: Binary array of violations (1 indicates violation)

        Returns:
            TBFI test statistic
        """
        term = 0
        last_failure_time = 0

        for i in range(1, matrix.size + 1):
            if matrix[i - 1] == 1:  # Found a violation
                if last_failure_time > 0:  # Not the first violation
                    n_i = i - last_failure_time  # Time between failures
                    # Calculate likelihood ratio term
                    numerator = self.theta * (1 - self.theta) ** (n_i - 1)
                    denominator = (1 / n_i) * (1 - 1 / n_i) ** (n_i - 1)
                    term += np.log(numerator / denominator)
                last_failure_time = i
        return -2 * term

    def _compute_hass_test(self) -> float:
        """
        Compute the Hass test statistic combining coverage and time-between-failures tests.

        Returns:
            Hass test statistic
        """
        assert self.var_violation_mtx is not None

        coverage_stat = self._compute_coverage_statistic()
        tbfi_stat = self._compute_time_between_failures(self.var_violation_mtx)
        return coverage_stat + tbfi_stat

    def _compute_christoffersen_test(self) -> float:
        """
        Compute the Christoffersen test statistic combining independence and coverage tests.

        Returns:
            Combined test statistic
        """
        assert self.var_violation_mtx is not None

        m_00, m_01, m_10, m_11 = self._count_transitions(self.var_violation_mtx)
        independence_stat = self._compute_independence_statistic(m_00, m_01, m_10, m_11)
        coverage_stat = self._compute_coverage_statistic()
        return independence_stat + coverage_stat

    def _compute_binomial_test(self) -> float:
        """
        Compute the binomial test statistic (Z-statistic).

        Returns:
            Z-statistic for the binomial test
        """
        assert self.var_violation_mtx is not None
        assert self.var_violations is not None

        test = scipy.stats.binomtest(
            self.var_violations, self.var_violation_mtx.size, self.theta
        )
        return test.pvalue

    def _run_bootstrap_loop(
        self, sample: NDArray[np.float64], quantile: float, num_bootstrap: int
    ) -> NDArray[np.float64]:
        """
        Run bootstrap iterations to compute test statistics.

        Args:
            sample: Array of observations
            quantile: Expected quantile value
            num_bootstrap: Number of bootstrap iterations

        Returns:
            Array of bootstrap test statistics
        """
        test_stats = np.zeros(num_bootstrap)
        bootstrap_size = max(len(sample) * 5, MIN_BOOTSTRAP_SIZE)

        for i in range(num_bootstrap):
            bootstrap_sample = np.random.choice(
                sample, size=bootstrap_size, replace=True
            )
            # Calculate bootstrap mean and standard error
            bootstrap_mean = np.mean(bootstrap_sample)
            bootstrap_std = np.std(bootstrap_sample)
            # Add small constant to avoid division by very small numbers
            bootstrap_sem = bootstrap_std / np.sqrt(bootstrap_size)
            test_stats[i] = (bootstrap_mean - quantile) / bootstrap_sem
        return test_stats

    def _compute_bootstrap_test(self) -> float:
        """
        Compute the bootstrap test p-value.

        Returns:
            Bootstrap test p-value
        """
        assert self.var_violation_mtx is not None

        bootstrap_stats = self._run_bootstrap_loop(
            self.var_violation_mtx.astype(np.float64), self.theta, BOOTSTRAP_ITERATIONS
        )
        num_significant = np.sum(
            (BOOTSTRAP_LOWER_BOUND <= bootstrap_stats)
            & (bootstrap_stats <= BOOTSTRAP_UPPER_BOUND)
        )
        return num_significant / bootstrap_stats.size

    def _compute_quantile_ratio(self) -> float:
        """
        Calculate the quantile ratio of the VaR and CVaR.

        Returns:
            The mean quantile ratio of the VaR and CVaR.
        """
        return float(np.mean(self.cvar / self.var))
