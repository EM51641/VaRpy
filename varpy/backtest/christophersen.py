from typing import Tuple

import numba as nb
import numpy as np
from numba import prange

BOOTSTRAP_UPPER_BOUND = 1.962
BOOTSTRAP_LOWER_BOUND = -1.962


@nb.njit()
def count_transitions(matrix: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Count the transitions between violations and non-violations in the matrix.

    Args:
        matrix: Binary array where 0 represents a violation and 1 represents no violation

    Returns:
        Tuple of (m_00, m_01, m_10, m_11) where:
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


def compute_christoffersen_test(
    matrix: np.ndarray, expected_prob: float, num_violations: int
) -> float:
    """
    Compute the Christoffersen test statistic combining independence and coverage tests.

    Args:
        matrix: Binary array of violations
        expected_prob: Expected probability of violation
        num_violations: Number of violations observed

    Returns:
        Combined test statistic
    """
    m_00, m_01, m_10, m_11 = count_transitions(matrix)
    independence_stat = compute_independence_statistic(m_00, m_01, m_10, m_11)
    coverage_stat = compute_coverage_statistic(
        num_violations, expected_prob, matrix.size
    )
    return independence_stat + coverage_stat


def compute_independence_statistic(m_00: int, m_01: int, m_10: int, m_11: int) -> float:
    """
    Compute the independence test statistic (CCI).

    Args:
        m_00: transitions from no violation to no violation
        m_01: transitions from no violation to violation
        m_10: transitions from violation to no violation
        m_11: transitions from violation to violation

    Returns:
        Independence test statistic
    """
    total = m_00 + m_01 + m_10 + m_11
    pi = (m_01 + m_11) / total
    pi1 = m_11 / (m_10 + m_11) if (m_10 + m_11) > 0 else 0
    pi0 = m_01 / (m_00 + m_01) if (m_00 + m_01) > 0 else 0

    term_a = 2 * (m_00 + m_01) * np.log(1 - pi) - 2 * (m_10 + m_11) * np.log(pi)
    term_b = (
        2 * m_00 * np.log(1 - pi0)
        + 2 * m_01 * np.log(pi0)
        + 2 * m_10 * np.log(1 - pi1)
        + 2 * m_11 * np.log(pi1)
    )
    return -term_a + term_b


def compute_coverage_statistic(
    num_violations: int, expected_prob: float, total_points: int
) -> float:
    """
    Compute the coverage test statistic (POF - Proportion of Failures).

    Args:
        num_violations: Number of violations observed
        expected_prob: Expected probability of violation
        total_points: Total number of observations

    Returns:
        Coverage test statistic
    """
    term_a = 2 * (total_points - num_violations) * np.log(
        1 - expected_prob
    ) + 2 * num_violations * np.log(expected_prob)
    term_b = 2 * (total_points - num_violations) * np.log(
        1 - (num_violations / total_points)
    ) + 2 * num_violations * np.log(num_violations / total_points)
    return -term_a + term_b


def compute_tuff_test(matrix: np.ndarray, expected_prob: float) -> float:
    """
    Compute the TUFF (Time Until First Failure) test statistic.

    Args:
        matrix: Binary array of violations
        num_violations: Number of violations observed
        expected_prob: Expected probability of violation

    Returns:
        TUFF test statistic
    """
    first_failure_time = find_first_failure(matrix)
    term_a = 2 * np.log(expected_prob) + (first_failure_time - 1) * np.log(
        1 - expected_prob
    )
    term_b = 2 * (1 / first_failure_time) + (first_failure_time - 1) * np.log(
        1 - 1 / first_failure_time
    )
    return -term_a + term_b


def find_first_failure(matrix: np.ndarray) -> int:
    """
    Find the index of the first violation in the matrix.

    Args:
        matrix: Binary array of violations

    Returns:
        Index of first violation
    """
    return np.where(matrix == 0)[0][0]


def compute_hass_test(
    matrix: np.ndarray, num_violations: int, expected_prob: float, total_points: int
) -> float:
    """
    Compute the Hass test statistic combining coverage and time-between-failures tests.

    Args:
        matrix: Binary array of violations
        num_violations: Number of violations observed
        expected_prob: Expected probability of violation
        total_points: Total number of observations

    Returns:
        Hass test statistic
    """
    coverage_stat = compute_coverage_statistic(
        num_violations, expected_prob, total_points
    )
    tbfi_stat = compute_time_between_failures(matrix, expected_prob)
    return coverage_stat + tbfi_stat


@nb.njit()
def compute_time_between_failures(matrix: np.ndarray, expected_prob: float) -> float:
    """
    Compute the Time Between Failures Independence (TBFI) test statistic.

    Args:
        matrix: Binary array of violations
        expected_prob: Expected probability of violation

    Returns:
        TBFI test statistic
    """
    term_a = term_b = 0
    last_failure_time = 0

    for i in range(matrix.size - 1):
        if matrix[i] == 0:
            time_since_last = i - last_failure_time
            term_a += 2 * np.log(expected_prob) + (time_since_last - 1) * np.log(
                1 - expected_prob
            )
            term_b += 2 * (1 / time_since_last) + (time_since_last - 1) * np.log(
                1 - 1 / time_since_last
            )
            last_failure_time = i

    return -term_a + term_b


def compute_bootstrap_test(
    sample: np.ndarray, quantile: float, num_bootstrap: int
) -> float:
    """
    Compute the bootstrap test p-value.

    Args:
        sample: Array of observations
        quantile: Expected quantile value
        num_bootstrap: Number of bootstrap iterations

    Returns:
        Bootstrap test p-value
    """
    bootstrap_stats = run_bootstrap_loop(sample, quantile, num_bootstrap)
    num_significant = np.sum(
        (BOOTSTRAP_LOWER_BOUND <= bootstrap_stats)
        & (bootstrap_stats <= BOOTSTRAP_UPPER_BOUND)
    )
    return num_significant / bootstrap_stats.size


@nb.njit(parallel=True, fastmath=True)
def run_bootstrap_loop(
    sample: np.ndarray, quantile: float, num_bootstrap: int
) -> np.ndarray:
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
    for i in prange(num_bootstrap):
        bootstrap_sample = np.random.choice(sample, size=252, replace=True)
        # Using median to adjust for skewness and kurtosis
        test_stats[i] = (np.median(bootstrap_sample) - quantile) / (
            np.std(bootstrap_sample) / np.sqrt(252)
        )
    return test_stats
