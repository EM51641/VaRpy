import numpy as np
import pytest
from numpy.typing import NDArray

from varpy.backtest.christophersen import BacktestResults


@pytest.fixture
def data() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Create test data for backtesting."""
    # Create returns with some violations
    ret = np.array([-0.02, -0.01, 0.01, 0.02, -0.03, 0.01, -0.02, 0.01, -0.01, 0.02])
    # VaR at 5% level
    var = np.full_like(ret, -0.01)
    # CVaR at 5% level (slightly more conservative)
    cvar = np.full_like(ret, -0.015)
    return ret, var, cvar


def test_backtest_results_initialization(
    data: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
):
    """Test initialization of BacktestResults class."""
    ret, var, cvar = data
    theta = 0.05

    results = BacktestResults(ret, var, cvar, theta)

    # Check initial state
    assert results.var_violations is None
    assert results.cvar_violations is None
    assert results.var_violation_mtx is None
    assert results.cvar_violation_mtx is None
    assert results.christoffersen is None
    assert results.binomial is None
    assert results.kupiec is None
    assert results.tuff is None
    assert results.hass is None
    assert results.q_ratio is None
    assert results.q_ratio_bootstrap is None


def test_violation_matrix_creation(
    data: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
):
    """Test creation of violation matrices."""
    ret, var, cvar = data
    results = BacktestResults(ret, var, cvar, 0.05)

    results.run()

    assert results.var_violation_mtx is not None
    assert results.cvar_violation_mtx is not None

    # Check matrix shapes
    assert results.var_violation_mtx.shape == ret.shape
    assert results.cvar_violation_mtx.shape == ret.shape

    # Check binary values
    assert np.all(np.isin(results.var_violation_mtx, [0, 1]))
    assert np.all(np.isin(results.cvar_violation_mtx, [0, 1]))

    # CVaR violations should be a subset of VaR violations
    assert np.all(results.cvar_violation_mtx <= results.var_violation_mtx)

    # Check if the violation matrices are correct
    assert np.array_equal(
        results.cvar_violation_mtx,
        np.where(ret < cvar, 1, 0),
    )
    assert np.array_equal(
        results.var_violation_mtx,
        np.where(ret < var, 1, 0),
    )


def test_violation_counts(
    data: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
):
    """Test counting of violations."""
    ret, var, cvar = data
    results = BacktestResults(ret, var, cvar, 0.05)

    results.run()

    # Counts should be non-negative
    assert results.var_violations is not None
    assert results.cvar_violations is not None

    assert results.var_violations >= 0
    assert results.cvar_violations >= 0

    # CVaR violations should be less than or equal to VaR violations
    assert results.cvar_violations <= results.var_violations


def test_binomial_test(
    data: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
):
    """Test binomial test computation."""
    ret, var, cvar = data
    results = BacktestResults(ret, var, cvar, 0.05)

    # Run computations
    results.run()

    assert results.binomial

    # Check p-value is between 0 and 1
    assert 0 <= results.binomial <= 1


def test_bootstrap_test(
    data: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
):
    """Test bootstrap test computation."""
    ret, var, cvar = data
    results = BacktestResults(ret, var, cvar, 0.05)

    # Run computations
    results.run()

    assert results.q_ratio_bootstrap is not None

    # Check p-value is between 0 and 1
    assert 0 <= results.q_ratio_bootstrap <= 1


def test_quantile_ratio(
    data: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
):
    """Test quantile ratio computation."""
    ret, var, cvar = data
    results = BacktestResults(ret, var, cvar, 0.05)

    results.run()

    assert results.q_ratio is not None

    # Ratio should be positive and greater than 1 (CVaR is more conservative)
    assert results.q_ratio > 1


def test_coverage_statistic(
    data: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
):
    """Test coverage statistic computation."""
    ret, var, cvar = data
    results = BacktestResults(ret, var, cvar, 0.05)

    results.run()

    assert results.kupiec is not None
    assert results.tuff is not None
    assert results.hass is not None

    assert results.kupiec > 0
    assert results.tuff > 0
    assert results.hass > 0


def test_christoffersen_test(
    data: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
):
    """Test Christoffersen test computation."""
    ret, var, cvar = data
    results = BacktestResults(ret, var, cvar, 0.05)

    results.run()

    assert results.christoffersen is not None
    assert results.christoffersen > 0
