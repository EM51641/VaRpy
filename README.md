# VaRpy - Value at Risk Models

A Python library for computing Value at Risk (VaR) and Conditional Value at Risk (CVaR) using various statistical distributions and GARCH models.

## Download

```bash
pip install varpy
```

## Recent Refactoring

The codebase has undergone a major refactoring to improve code organization, maintainability, and consistency. Here are the key changes:

### 1. Standardized Model Structure

Each VaR model (Normal, Student's t, EVT) follows the same pattern:

- Constructor takes only `theta` and `horizon` parameters
- `run` method accepts returns data as input
- Private helper methods for specific computations
- Consistent method naming and documentation

Example structure:
```python
class SomeVar(BaseVar):
    def __init__(self, theta: float, horizon: int):
        super().__init__(theta, horizon)

    def run(self, ret: NDArray[np.float64]) -> None:
        # 1. Get GARCH forecasts
        # 2. Process innovations
        # 3. Fit distribution
        # 4. Compute VaR/CVaR
        pass
```

### 3. Improved Type Hints

- Added comprehensive type hints using `numpy.typing`
- Consistent use of `NDArray[np.float64]` for numpy arrays
- Proper return type annotations for all methods

### 5. Enhanced Documentation

- Comprehensive docstrings for all classes and methods
- Clear parameter and return type documentation
- Consistent documentation style across all models

### 6. Code Organization

- Moved models to `varpy/var/models/` directory
- Consistent file naming (lowercase with underscores)
- Clear separation of concerns between models

## Available Models

### Normal Distribution
- Assumes returns follow a normal distribution
- Uses GARCH(1,1) with Gaussian innovations
- Suitable for well-behaved financial returns

### Student's t Distribution
- Assumes returns follow a Student's t distribution
- Uses GARCH(1,1) with Student's t innovations
- Better suited for heavy-tailed returns

### Extreme Value Theory (EVT)
- Uses Generalized Pareto Distribution for tail modeling
- Combines GARCH with EVT for better tail estimation
- Most suitable for extreme risk measurement

## Usage Example

```python
import numpy as np
from varpy.var.models import Normal, Student, EVT

# Initialize model
model = Normal(theta=0.05, horizon=1)

# Compute VaR and CVaR
returns = np.array([...])  # Your return data
model.run(returns)

# Get results
var = model.var
cvar = model.cvar
```

More exemples [here](exemples.ipynb)

## Dependencies

- numpy
- scipy
- arch (for GARCH modeling)

## Contributing

When adding new VaR models, please follow the established pattern:
1. Inherit from `BaseVar`
2. Implement the `run` method
3. Use private helper methods for specific computations
4. Add comprehensive type hints and documentation
5. Follow the modular design pattern 