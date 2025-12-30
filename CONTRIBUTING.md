# Contributing to KAN-MLX-Physics

Thank you for your interest in contributing to KAN-MLX-Physics! This guide will help you get started.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Style](#code-style)
4. [Testing](#testing)
5. [Pull Request Process](#pull-request-process)
6. [Project Structure](#project-structure)
7. [Adding Features](#adding-features)

---

## Getting Started

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- Git

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yaelmartinez/kan-mlx-physics.git
cd kan-mlx-physics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "from kan_mlx_physics import MultKAN; print('Success!')"
```

---

## Development Setup

### Dependencies

The project uses the following key dependencies:

| Package | Purpose |
|---------|---------|
| `mlx` | Apple's ML framework |
| `numpy` | Numerical operations |
| `scipy` | LBFGS optimizer, special functions |
| `matplotlib` | Visualization |
| `sympy` | Symbolic mathematics |

### Installing Development Dependencies

```bash
# Full development installation
pip install -e ".[dev]"

# This includes:
# - pytest (testing)
# - black (formatting)
# - ruff (linting)
# - mypy (type checking)
```

### Editor Setup

**VS Code:**
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true
}
```

**PyCharm:**
- Set interpreter to `.venv/bin/python`
- Enable Black formatter
- Enable Ruff linter

---

## Code Style

### Formatting

We use **Black** for consistent formatting:

```bash
# Format all files
black src/kan_mlx_physics/

# Check without modifying
black --check src/kan_mlx_physics/
```

Configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 100
target-version = ['py310']
```

### Linting

We use **Ruff** for fast linting:

```bash
# Run linter
ruff check src/kan_mlx_physics/

# Auto-fix issues
ruff check --fix src/kan_mlx_physics/
```

### Type Hints

Use type hints for public functions:

```python
# Good
def create_dataset(
    f: Callable[[mx.array], mx.array],
    n_var: int = 2,
    train_num: int = 1000,
) -> Dict[str, mx.array]:
    ...

# Avoid
def create_dataset(f, n_var=2, train_num=1000):
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def fit_affine_params(
    x: np.ndarray,
    y: np.ndarray,
    fn_name: str,
) -> Tuple[float, float, float, float, float]:
    """Fit affine parameters for symbolic function.

    Finds optimal (a, b, c, d) such that y ≈ c * f(a*x + b) + d.

    Args:
        x: Input values, shape (N,).
        y: Target values, shape (N,).
        fn_name: Name of symbolic function from registry.

    Returns:
        Tuple of (a, b, c, d, r2) where r2 is the R-squared score.

    Raises:
        ValueError: If fn_name is not in the registry.

    Example:
        >>> a, b, c, d, r2 = fit_affine_params(x, y, 'sin')
        >>> print(f"y ≈ {c:.2f} * sin({a:.2f}*x + {b:.2f}) + {d:.2f}")
    """
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_multkan.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=kan_mlx_physics --cov-report=html
```

### Writing Tests

Tests go in `tests/` directory:

```python
# tests/test_spline.py
import pytest
import mlx.core as mx
from kan_mlx_physics.spline import B_batch, coef2curve

class TestBSpline:
    def test_B_batch_shape(self):
        """Test that B_batch returns correct shape."""
        x = mx.linspace(0, 1, 10)
        grid = mx.linspace(0, 1, 8)
        k = 3

        basis = B_batch(x, grid, k)

        assert basis.shape == (10, 5)  # 8 - k = 5 basis functions

    def test_B_batch_partition_of_unity(self):
        """Test that basis functions sum to 1."""
        x = mx.linspace(0.1, 0.9, 100)
        grid = mx.linspace(0, 1, 10)
        k = 3

        basis = B_batch(x, grid, k)
        sums = mx.sum(basis, axis=1)

        assert mx.allclose(sums, mx.ones_like(sums), atol=1e-5)
```

### Test Categories

| Category | Description |
|----------|-------------|
| Unit tests | Test individual functions |
| Integration tests | Test component interactions |
| Regression tests | Ensure bugs don't return |
| Performance tests | Benchmark critical paths |

---

## Pull Request Process

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes

- Write code following our style guide
- Add tests for new functionality
- Update documentation if needed

### 3. Run Quality Checks

```bash
# Format
black src/kan_mlx_physics/

# Lint
ruff check src/kan_mlx_physics/

# Test
pytest

# Type check (optional but appreciated)
mypy src/kan_mlx_physics/
```

### 4. Commit

Use clear commit messages:

```bash
# Good
git commit -m "Add LBFGS optimizer to functional module"
git commit -m "Fix NaN handling in symbolic regression"

# Avoid
git commit -m "Update"
git commit -m "Fix bug"
```

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear description of changes
- Link to related issue (if any)
- Test results

### 6. Review

- Address reviewer feedback
- Keep commits clean (squash if needed)
- Ensure CI passes

---

## Project Structure

```
kan-mlx-physics/
├── src/
│   └── kan_mlx_physics/
│       ├── __init__.py          # Public API exports
│       ├── multkan.py           # MultKAN model class
│       ├── kan_layer.py         # KANLayer class
│       ├── spline.py            # B-spline functions
│       ├── symbolic.py          # Symbolic registry & regression
│       ├── physics_symbolic.py  # Physics functions
│       ├── functional.py        # Pure functional API
│       ├── pinn.py              # PINN utilities
│       ├── visualization.py     # Plotting
│       ├── formula_render.py    # Formula output
│       ├── utils.py             # Dataset utilities
│       └── pde/                 # PDE solver module
│           ├── __init__.py
│           ├── dsl.py           # Equation parser
│           ├── operators.py     # Differential operators
│           ├── physics.py       # Physics problems
│           └── solver.py        # Solver config
├── tests/
│   ├── test_multkan.py
│   ├── test_spline.py
│   ├── test_symbolic.py
│   └── ...
├── docs/
│   ├── QUICKSTART.md
│   ├── API_REFERENCE.md
│   ├── PHYSICS_GUIDE.md
│   └── PYKAN_MIGRATION.md
├── benchmarks/
│   └── benchmark.py             # Reproducible benchmarks
├── README.md
├── CONTRIBUTING.md
├── LICENSE
└── pyproject.toml
```

---

## Adding Features

### Adding a New Symbolic Function

1. **Edit `symbolic.py`:**

```python
# In SYMBOLIC_REGISTRY or register_symbolic()
register_symbolic(
    name="my_function",
    fn=lambda x: mx.sin(x) * mx.exp(-x),  # MLX version
    fn_np=lambda x: np.sin(x) * np.exp(-x),  # NumPy version
    latex=r"\sin(x) e^{-x}",
    typst="sin(x) e^(-x)",
    complexity=3,
)
```

2. **Add tests:**

```python
# tests/test_symbolic.py
def test_my_function():
    from kan_mlx_physics.symbolic import SYMBOLIC_REGISTRY
    fn = SYMBOLIC_REGISTRY['my_function']

    x = mx.array([0.0, 1.0, 2.0])
    y = fn.fn(x)

    expected = mx.sin(x) * mx.exp(-x)
    assert mx.allclose(y, expected)
```

3. **Update `__init__.py` if public.**

### Adding a New PDE Problem

1. **Edit `pde/physics.py`:**

```python
@dataclass
class MyNewEquation:
    """Description of the equation."""

    param1: float = 1.0
    param2: float = 0.0

    def residual(self, u: Callable, x: mx.array) -> mx.array:
        """Compute PDE residual."""
        # Implement your equation
        return ...

    def boundary_conditions(self) -> List[BoundaryCondition]:
        """Define boundary conditions."""
        return [
            BoundaryCondition.dirichlet(0.0, where=lambda x: x[:, 0] < -4.9),
            ...
        ]
```

2. **Register in `pde/dsl.py` templates.**

3. **Add documentation and tests.**

### Adding a New Optimizer

1. **Edit `functional.py`:**

```python
def my_optimizer_update(
    params: ParamsList,
    grads: ParamsList,
    state: Any,
    lr: float = 0.001,
    **kwargs,
) -> Tuple[ParamsList, Any]:
    """My custom optimizer step.

    Args:
        params: Current parameters.
        grads: Gradients.
        state: Optimizer state.
        lr: Learning rate.

    Returns:
        Updated (params, state).
    """
    # Implementation
    return new_params, new_state
```

2. **Update `MultKAN.fit()` to support new optimizer.**

3. **Add tests and documentation.**

---

## Questions?

- Open an issue for bugs or feature requests
- Discussions for general questions
- Email for sensitive matters

Thank you for contributing to KAN-MLX-Physics!
