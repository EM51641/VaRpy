import scipy


def binomial_test(violations: int, n: int, theta: float) -> float:
    """
    Perform a binomial test to determine if the number of violations is statistically significant.

    Args:
        violations: The number of violations observed.
        n: The total number of observations.
        theta: The probability of a violation.

    Returns:
        The p-value of the binomial test.
    """
    test = scipy.stats.binomtest(violations, n, theta)
    return test.pvalue
