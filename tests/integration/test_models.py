import numpy as np
import pytest
from numpy.typing import NDArray

from varpy.models.evt import EVT
from varpy.models.normal import Normal
from varpy.models.student import Student


@pytest.fixture
def sample_returns() -> NDArray[np.float64]:
    """Generate sample returns for testing."""
    np.random.seed(42)
    return np.random.normal(0, 1, 1000)


@pytest.fixture
def extreme_returns() -> NDArray[np.float64]:
    """Generate returns with extreme values for EVT testing."""
    np.random.seed(42)
    normal_returns = np.random.normal(0, 1, 1000)
    # Add some extreme values
    extreme_returns = np.concatenate([normal_returns, np.random.normal(0, 5, 100)])
    return extreme_returns


def test_normal_var_model(sample_returns: NDArray[np.float64]) -> None:
    """Test Normal VaR model."""
    model = Normal(theta=0.05, horizon=1)
    model.run(sample_returns)

    # Test VaR and CVaR are computed
    assert model.var is not None
    assert model.cvar is not None

    # Test VaR is more conservative than CVaR
    assert abs(model.var) < abs(model.cvar)

    # Test values are reasonable (within expected range for normal distribution)
    assert -5 < model.var < 0
    assert -10 < model.cvar < 0
    assert model.var > model.cvar


def test_student_var_model(sample_returns: NDArray[np.float64]) -> None:
    """Test Student's t VaR model."""
    model = Student(theta=0.05, horizon=1)
    model.run(sample_returns)

    # Test VaR and CVaR are computed
    assert model.var is not None
    assert model.cvar is not None

    # Test VaR is more conservative than CVaR
    assert abs(model.var) < abs(model.cvar)

    # Test values are reasonable (should be more conservative than normal)
    assert -5 < model.var < 0
    assert -10 < model.cvar < 0
    assert model.var > model.cvar


def test_evt_var_model(extreme_returns: NDArray[np.float64]) -> None:
    """Test EVT VaR model."""
    model = EVT(theta=0.05, horizon=1)
    model.run(extreme_returns)

    # Test VaR and CVaR are computed
    assert model.var is not None
    assert model.cvar is not None

    # Test VaR is more conservative than CVaR
    assert abs(model.var) < abs(model.cvar)

    # Test values are reasonable (should be more conservative than normal/student)
    assert -10 < model.var < 0
    assert -15 < model.cvar < 0

    assert model.var > model.cvar


def test_different_theta_values(sample_returns: NDArray[np.float64]) -> None:
    """Test models with different theta values."""
    theta_values = [0.01, 0.10]
    models = [Normal, Student, EVT]

    for theta in theta_values:
        for model_class in models:
            model = model_class(theta=theta, horizon=1)  # type: ignore
            model.run(sample_returns)

            # Test VaR and CVaR are computed
            assert model.var is not None
            assert model.cvar is not None

            # Test that more extreme theta (smaller) leads to more conservative VaR
            if theta == 0.01:
                assert abs(model.var) > 1.5  # Should be more conservative
            elif theta == 0.10:
                assert abs(model.var) < 1.5  # Should be less conservative


def test_different_horizons(sample_returns: NDArray[np.float64]) -> None:
    """Test models with different forecast horizons."""
    horizons = [1, 5, 10]
    models = [Normal, Student, EVT]

    for horizon in horizons:
        for model_class in models:
            model = model_class(theta=0.05, horizon=horizon)  # type: ignore
            model.run(sample_returns)

            # Test VaR and CVaR are computed
            assert model.var is not None
            assert model.cvar is not None

            # Test that longer horizons lead to more conservative VaR
            if horizon > 1:
                assert abs(model.var) > 1.0  # Should be more conservative


def test_invalid_inputs() -> None:
    """Test models with invalid inputs."""
    # Test invalid theta
    with pytest.raises(ValueError):
        Normal(theta=1.5, horizon=1)  # theta should be between 0 and 1

    with pytest.raises(ValueError):
        Student(theta=-0.1, horizon=1)  # theta should be positive

    # Test invalid horizon
    with pytest.raises(ValueError):
        EVT(theta=0.05, horizon=0)  # horizon should be positive

    # Test empty returns array
    empty_returns = np.array([])
    model = Normal(theta=0.05, horizon=1)
    with pytest.raises(ValueError):
        model.run(empty_returns)
