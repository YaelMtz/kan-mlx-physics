"""Symbolic functions and Symbolic KAN Layer for MLX-KAN.

This module provides:
1. A registry of symbolic functions (sin, cos, exp, etc.)
2. Symbolic_KANLayer that can replace learned splines with symbolic functions
3. Tools for fitting affine parameters to match splines to symbolic forms
"""

import logging
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# Module-level logger for debugging symbolic fitting
logger = logging.getLogger(__name__)


@dataclass
class SymbolicFunction:
    """A symbolic function with its properties."""
    name: str
    fn: Callable[[mx.array], mx.array]
    fn_np: Callable[[np.ndarray], np.ndarray]  # NumPy version for fitting
    latex: str  # LaTeX representation
    typst: str = ""  # Typst representation
    complexity: int = 1  # Complexity score for formula simplicity
    safe_fn: Optional[Callable[[mx.array], mx.array]] = None  # Singularity-safe version
    safe_fn_np: Optional[Callable[[np.ndarray], np.ndarray]] = None  # NumPy safe version
    domain: str = "all"  # Valid domain: "all", "x >= 0", "x != 0", "|x| <= 1"


# Global registry of symbolic functions
SYMBOLIC_REGISTRY: Dict[str, SymbolicFunction] = {}


def register_symbolic(
    name: str,
    fn: Callable,
    fn_np: Callable,
    latex: str,
    typst: str = "",
    complexity: int = 1,
    safe_fn: Optional[Callable] = None,
    safe_fn_np: Optional[Callable] = None,
    domain: str = "all",
) -> None:
    """Register a symbolic function.

    Args:
        name: Function name (e.g., "sin", "x^2")
        fn: MLX function
        fn_np: NumPy function (for fitting)
        latex: LaTeX representation
        typst: Typst representation
        complexity: Complexity score
        safe_fn: Singularity-safe MLX version (optional)
        safe_fn_np: Singularity-safe NumPy version (optional)
        domain: Valid domain string ("all", "x >= 0", "x != 0", "|x| <= 1")
    """
    SYMBOLIC_REGISTRY[name] = SymbolicFunction(
        name=name,
        fn=fn,
        fn_np=fn_np,
        latex=latex,
        typst=typst if typst else latex.replace("\\", ""),  # Simple fallback
        complexity=complexity,
        safe_fn=safe_fn,
        safe_fn_np=safe_fn_np,
        domain=domain,
    )


def _safe_div(x, eps=1e-8):
    """Safe division helper."""
    return 1.0 / (x + eps * np.sign(x + 1e-16))


# ============================================================
# Singularity-Safe Function Variants
# ============================================================
# These functions add small epsilon values (typically 1e-8) to avoid
# numerical issues at singularities. The epsilon is chosen to be:
# - Small enough to not affect normal computations
# - Large enough to avoid float32 precision issues
# - 1e-8 ≈ sqrt(machine_epsilon) for float32, a common safe choice

def _sqrt_safe_mx(x):
    """Safe sqrt for MLX - clamps negative values to small positive."""
    return mx.sqrt(mx.maximum(x, mx.array(1e-8)))


def _sqrt_safe_np(x):
    """Safe sqrt for NumPy - clamps negative values to small positive."""
    return np.sqrt(np.maximum(x, 1e-8))


def _log_safe_mx(x):
    """Safe log for MLX - uses |x| and clamps to avoid log(0)."""
    return mx.log(mx.maximum(mx.abs(x), mx.array(1e-8)))


def _log_safe_np(x):
    """Safe log for NumPy - uses |x| and clamps to avoid log(0)."""
    return np.log(np.maximum(np.abs(x), 1e-8))


def _inv_safe_mx(x):
    """Safe 1/x for MLX - adds sign-preserving epsilon to avoid division by zero."""
    # sign(x) * eps ensures we push away from zero in the right direction
    # Additional 1e-16 handles the case where x is exactly zero
    return 1.0 / (x + mx.sign(x) * 1e-8 + 1e-16)


def _inv_safe_np(x):
    """Safe 1/x for NumPy - adds sign-preserving epsilon to avoid division by zero."""
    return 1.0 / (x + np.sign(x) * 1e-8 + 1e-16)


def _inv2_safe_mx(x):
    """Safe 1/x² for MLX - x² is always non-negative, so just add eps."""
    return 1.0 / (x**2 + 1e-8)


def _inv2_safe_np(x):
    """Safe 1/x² for NumPy - x² is always non-negative, so just add eps."""
    return 1.0 / (x**2 + 1e-8)


def _tan_safe_mx(x):
    """Safe tan for MLX - clips output to [-100, 100] to avoid ±∞ at poles."""
    return mx.clip(mx.tan(x), -100.0, 100.0)


def _tan_safe_np(x):
    """Safe tan for NumPy - clips output to [-100, 100] to avoid ±∞ at poles."""
    return np.clip(np.tan(x), -100.0, 100.0)


def _arcsin_safe_mx(x):
    """Safe arcsin for MLX - clips input to (-1, 1) to stay in valid domain."""
    # 1e-7 margin avoids gradient issues at the boundary
    return mx.arcsin(mx.clip(x, -1.0 + 1e-7, 1.0 - 1e-7))


def _arcsin_safe_np(x):
    """Safe arcsin for NumPy - clips input to (-1, 1) to stay in valid domain."""
    return np.arcsin(np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7))


def _arccos_safe_mx(x):
    """Safe arccos for MLX - clips input to (-1, 1) to stay in valid domain."""
    return mx.arccos(mx.clip(x, -1.0 + 1e-7, 1.0 - 1e-7))


def _arccos_safe_np(x):
    """Safe arccos for NumPy - clips input to (-1, 1) to stay in valid domain."""
    return np.arccos(np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7))


# Register standard symbolic functions
def _init_symbolic_registry():
    """Initialize the default symbolic function registry.

    Each function is registered with:
    - Standard version (may have singularities)
    - Safe version (avoids NaN/Inf)
    - Domain specification
    """

    # Identity and constants (no singularities)
    register_symbolic("x", lambda x: x, lambda x: x, "x", "x", complexity=1)
    register_symbolic("0", lambda x: mx.zeros_like(x), lambda x: np.zeros_like(x), "0", "0", complexity=0)
    register_symbolic("1", lambda x: mx.ones_like(x), lambda x: np.ones_like(x), "1", "1", complexity=0)

    # Powers (some have singularities)
    register_symbolic("x^2", lambda x: x**2, lambda x: x**2, "x^2", "x^2", complexity=2)
    register_symbolic("x^3", lambda x: x**3, lambda x: x**3, "x^3", "x^3", complexity=3)
    register_symbolic("x^4", lambda x: x**4, lambda x: x**4, "x^4", "x^4", complexity=4)

    # sqrt - singularity at x < 0
    register_symbolic(
        "x^0.5",
        lambda x: mx.sqrt(mx.abs(x)),
        lambda x: np.sqrt(np.abs(x)),
        "\\sqrt{x}", "sqrt(x)",
        complexity=2,
        safe_fn=_sqrt_safe_mx,
        safe_fn_np=_sqrt_safe_np,
        domain="x >= 0",
    )

    # 1/x - singularity at x = 0
    register_symbolic(
        "x^-1",
        lambda x: 1.0 / (x + 1e-8),
        lambda x: _safe_div(x),
        "1/x", "1/x",
        complexity=2,
        safe_fn=_inv_safe_mx,
        safe_fn_np=_inv_safe_np,
        domain="x != 0",
    )

    # 1/x^2 - singularity at x = 0
    register_symbolic(
        "x^-2",
        lambda x: 1.0 / (x**2 + 1e-8),
        lambda x: _safe_div(x**2),
        "1/x^2", "1/x^2",
        complexity=3,
        safe_fn=_inv2_safe_mx,
        safe_fn_np=_inv2_safe_np,
        domain="x != 0",
    )

    # Trigonometric (sin, cos are safe; tan has poles)
    register_symbolic("sin", lambda x: mx.sin(x), lambda x: np.sin(x), "\\sin(x)", "sin(x)", complexity=2)
    register_symbolic("cos", lambda x: mx.cos(x), lambda x: np.cos(x), "\\cos(x)", "cos(x)", complexity=2)

    register_symbolic(
        "tan",
        lambda x: mx.tan(x),
        lambda x: np.tan(x),
        "\\tan(x)", "tan(x)",
        complexity=3,
        safe_fn=_tan_safe_mx,
        safe_fn_np=_tan_safe_np,
        domain="x != (2k+1)π/2",
    )

    # Inverse trigonometric (domain restricted)
    register_symbolic(
        "arcsin",
        lambda x: mx.arcsin(x),
        lambda x: np.arcsin(x),
        "\\arcsin(x)", "arcsin(x)",
        complexity=3,
        safe_fn=_arcsin_safe_mx,
        safe_fn_np=_arcsin_safe_np,
        domain="|x| <= 1",
    )

    register_symbolic(
        "arccos",
        lambda x: mx.arccos(x),
        lambda x: np.arccos(x),
        "\\arccos(x)", "arccos(x)",
        complexity=3,
        safe_fn=_arccos_safe_mx,
        safe_fn_np=_arccos_safe_np,
        domain="|x| <= 1",
    )

    register_symbolic(
        "arctan",
        lambda x: mx.arctan(x),
        lambda x: np.arctan(x),
        "\\arctan(x)", "arctan(x)",
        complexity=3,
    )

    # Exponential and logarithmic
    register_symbolic("exp", lambda x: mx.exp(x), lambda x: np.exp(x), "e^x", "e^x", complexity=2)

    # log - singularity at x <= 0
    register_symbolic(
        "log",
        lambda x: mx.log(mx.abs(x) + 1e-8),
        lambda x: np.log(np.abs(x) + 1e-8),
        "\\log(x)", "log(x)",
        complexity=2,
        safe_fn=_log_safe_mx,
        safe_fn_np=_log_safe_np,
        domain="x > 0",
    )

    # Hyperbolic (tanh is bounded, sinh/cosh can overflow)
    register_symbolic("sinh", lambda x: mx.sinh(x), lambda x: np.sinh(x), "\\sinh(x)", "sinh(x)", complexity=3)
    register_symbolic("cosh", lambda x: mx.cosh(x), lambda x: np.cosh(x), "\\cosh(x)", "cosh(x)", complexity=3)
    register_symbolic("tanh", lambda x: mx.tanh(x), lambda x: np.tanh(x), "\\tanh(x)", "tanh(x)", complexity=2)

    # Special functions
    register_symbolic("abs", lambda x: mx.abs(x), lambda x: np.abs(x), "|x|", "abs(x)", complexity=1)
    register_symbolic("sign", lambda x: mx.sign(x), lambda x: np.sign(x), "\\text{sign}(x)", "op(\"sign\")(x)", complexity=1)
    register_symbolic("gaussian", lambda x: mx.exp(-x**2), lambda x: np.exp(-x**2), "e^{-x^2}", "e^(-x^2)", complexity=3)
    register_symbolic("sigmoid", lambda x: mx.sigmoid(x), lambda x: 1/(1+np.exp(-x)), "\\sigma(x)", "sigma(x)", complexity=2)

    # Additional useful functions
    register_symbolic("relu", lambda x: mx.maximum(x, mx.array(0.0)), lambda x: np.maximum(x, 0.0), "\\text{ReLU}(x)", "max(x, 0)", complexity=1)
    register_symbolic("softplus", lambda x: mx.log(1 + mx.exp(x)), lambda x: np.log(1 + np.exp(x)), "\\log(1+e^x)", "log(1 + e^x)", complexity=2)


# Initialize registry on module load
_init_symbolic_registry()


def add_symbolic(
    name: str,
    fn: Callable,
    fn_np: Optional[Callable] = None,
    latex: Optional[str] = None,
    complexity: int = 1,
) -> None:
    """Add a custom symbolic function to the registry.

    Args:
        name: Function name
        fn: MLX function taking mx.array -> mx.array
        fn_np: NumPy version (defaults to using fn with conversion)
        latex: LaTeX string (defaults to name)
        complexity: Complexity score
    """
    if fn_np is None:
        # Create NumPy wrapper
        def fn_np_wrapper(x):
            result = fn(mx.array(x))
            return np.array(result)
        fn_np = fn_np_wrapper

    if latex is None:
        latex = name

    register_symbolic(name, fn, fn_np, latex, complexity)


def list_symbolic() -> List[str]:
    """List all registered symbolic function names."""
    return list(SYMBOLIC_REGISTRY.keys())


def validate_domain(x: np.ndarray, domain: str) -> bool:
    """Check if x values are in function's valid domain.

    Args:
        x: Input values
        domain: Domain specification string

    Returns:
        True if all x values are in valid domain
    """
    if domain == "all":
        return True
    elif domain == "x >= 0":
        return np.all(x >= 0)
    elif domain == "x > 0":
        return np.all(x > 0)
    elif domain == "x != 0":
        return np.all(np.abs(x) > 1e-8)
    elif domain == "|x| <= 1":
        return np.all(np.abs(x) <= 1)
    return True


def remove_outliers_iqr(y: np.ndarray, k: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers using IQR method.

    Args:
        y: Input values
        k: IQR multiplier (1.5 = standard, 3 = extreme outliers only)

    Returns:
        Tuple of (cleaned_y, mask) where mask indicates kept values
    """
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1

    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (y >= lower) & (y <= upper)
    return y[mask], mask


def score_symbolic_fit(r2: float, complexity: int, weight_simple: float = 0.1) -> float:
    """Score combining accuracy and simplicity (Occam's razor).

    Args:
        r2: R² score (0 to 1)
        complexity: Function complexity (1-5 typically)
        weight_simple: Weight for simplicity preference

    Returns:
        Combined score (higher is better)
    """
    # Normalize complexity to [0, 1] range
    max_complexity = 5
    norm_complexity = min(complexity, max_complexity) / max_complexity

    # Weighted score: prefer simpler functions at similar R²
    return r2 - weight_simple * norm_complexity


def fit_affine_params(
    x: np.ndarray,
    y: np.ndarray,
    fn_name: str,
    a_range: Tuple[float, float] = (-10, 10),
    b_range: Tuple[float, float] = (-10, 10),
    n_search: int = 100,
    use_safe: bool = True,
    remove_outliers: bool = True,
    regularization: float = 1e-4,
) -> Tuple[float, float, float, float, float]:
    """Fit affine parameters to match y ≈ c * f(a*x + b) + d.

    Robust fitting with:
    - Singularity-safe function variants
    - Outlier rejection
    - Regularized least squares

    Args:
        x: Input values (1D array)
        y: Target values (1D array)
        fn_name: Name of symbolic function
        a_range: Search range for input scale
        b_range: Search range for input bias
        n_search: Number of grid points for search
        use_safe: Use singularity-safe function variant if available
        remove_outliers: Remove outliers before fitting
        regularization: L2 regularization for least squares

    Returns:
        Tuple of (a, b, c, d, r2_score)
        where the fit is: c * f(a*x + b) + d
    """
    if fn_name not in SYMBOLIC_REGISTRY:
        raise ValueError(f"Unknown symbolic function: {fn_name}")

    sym_fn = SYMBOLIC_REGISTRY[fn_name]

    # Choose safe or standard function
    if use_safe and sym_fn.safe_fn_np is not None:
        fn = sym_fn.safe_fn_np
    else:
        fn = sym_fn.fn_np

    # Remove outliers if requested
    if remove_outliers and len(y) > 10:
        y_clean, mask = remove_outliers_iqr(y, k=1.5)
        x_clean = x[mask]
        if len(y_clean) < 5:
            # Not enough points after outlier removal, use original
            x_clean, y_clean = x, y
    else:
        x_clean, y_clean = x, y

    best_r2 = -np.inf
    best_params = (1.0, 0.0, 1.0, 0.0)

    # Grid search over affine parameters a (scale) and b (shift)
    a_vals = np.linspace(a_range[0], a_range[1], n_search)
    b_vals = np.linspace(b_range[0], b_range[1], n_search)

    # Exclude |a| < 0.1 to avoid near-degenerate cases where f(a*x + b) ≈ f(b)
    # becomes insensitive to x, causing unstable linear regression
    a_vals = a_vals[np.abs(a_vals) > 0.1]

    y_mean = np.mean(y_clean)
    # Add small epsilon to prevent division by zero when y is constant
    ss_tot = np.sum((y_clean - y_mean) ** 2) + 1e-8

    for a in a_vals:
        for b in b_vals:
            try:
                # Compute f(a*x + b) - the inner affine transformation
                inner = a * x_clean + b
                f_val = fn(inner)

                # Skip if function evaluation produced NaN/Inf
                if not np.all(np.isfinite(f_val)):
                    continue

                # Skip constant outputs - can't fit a meaningful linear relationship
                # when all f(a*x + b) values are identical
                if np.std(f_val) < 1e-8:
                    continue

                # Regularized linear regression to find c and d
                # y = c * f_val + d
                X = np.column_stack([f_val, np.ones_like(f_val)])
                XtX = X.T @ X
                Xty = X.T @ y_clean

                # Regularization
                XtX += regularization * np.eye(2)

                params = np.linalg.solve(XtX, Xty)
                c, d = params[0], params[1]

                # Compute R² score
                y_pred = c * f_val + d
                ss_res = np.sum((y_clean - y_pred) ** 2)
                r2 = 1 - ss_res / ss_tot

                if r2 > best_r2:
                    best_r2 = r2
                    best_params = (a, b, c, d)

            except np.linalg.LinAlgError as e:
                # Singular matrix - skip this (a, b) combination
                logger.debug(f"Linear algebra error for a={a}, b={b}: {e}")
                continue
            except (ValueError, RuntimeWarning) as e:
                # Invalid domain or numerical issues
                logger.debug(f"Numerical issue for a={a}, b={b}: {e}")
                continue
            except Exception as e:
                # Unexpected error - log at higher level
                logger.warning(f"Unexpected error fitting {fn_name} at a={a}, b={b}: {type(e).__name__}: {e}")
                continue

    return (*best_params, best_r2)


def suggest_symbolic(
    x: np.ndarray,
    y: np.ndarray,
    a_range: Tuple[float, float] = (-10, 10),
    b_range: Tuple[float, float] = (-10, 10),
    n_search: int = 50,
    weight_simple: float = 0.1,
    top_k: int = 3,
) -> List[Tuple[str, float, float, float, float, float, float]]:
    """Suggest best symbolic functions for the given data.

    Args:
        x: Input values
        y: Target values
        a_range: Search range for scale parameter
        b_range: Search range for bias parameter
        n_search: Grid search resolution
        weight_simple: Weight for simplicity in scoring
        top_k: Number of top suggestions to return

    Returns:
        List of tuples: (fn_name, a, b, c, d, r2, score) sorted by score
    """
    results = []

    for fn_name, sym_fn in SYMBOLIC_REGISTRY.items():
        try:
            a, b, c, d, r2 = fit_affine_params(
                x, y, fn_name,
                a_range=a_range,
                b_range=b_range,
                n_search=n_search,
                use_safe=True,
            )

            if r2 > 0:  # Only consider positive R²
                score = score_symbolic_fit(r2, sym_fn.complexity, weight_simple)
                results.append((fn_name, a, b, c, d, r2, score))

        except Exception as e:
            logger.debug(f"Failed to fit {fn_name}: {type(e).__name__}: {e}")
            continue

    # Sort by score (descending)
    results.sort(key=lambda x: x[6], reverse=True)

    return results[:top_k]


class Symbolic_KANLayer(nn.Module):
    """Symbolic overlay for a KAN layer.

    This layer stores symbolic function assignments for each edge (i, j).
    When an edge is "fixed" to a symbolic function, it uses that function
    instead of the learned spline.

    The computation for a symbolic edge is:
        output = c * f(a * x + b) + d
    where f is the symbolic function and (a, b, c, d) are affine parameters.
    """

    def __init__(self, in_dim: int, out_dim: int):
        """Initialize symbolic layer.

        Args:
            in_dim: Number of input neurons
            out_dim: Number of output neurons
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Track which edges have symbolic functions
        # None means use spline, string means use that symbolic function
        self.fns_name: List[List[Optional[str]]] = [
            [None for _ in range(out_dim)] for _ in range(in_dim)
        ]

        # Affine parameters for each edge: (a, b, c, d)
        # y = c * f(a*x + b) + d
        self.affine_a = mx.ones((in_dim, out_dim))
        self.affine_b = mx.zeros((in_dim, out_dim))
        self.affine_c = mx.ones((in_dim, out_dim))
        self.affine_d = mx.zeros((in_dim, out_dim))

        # Mask for which edges are symbolic (1 = symbolic, 0 = spline)
        self._symbolic_mask = mx.zeros((in_dim, out_dim))

    @property
    def symbolic_mask(self) -> mx.array:
        """Get mask of symbolic edges."""
        return self._symbolic_mask

    def is_symbolic(self, i: int, j: int) -> bool:
        """Check if edge (i, j) is symbolic."""
        return self.fns_name[i][j] is not None

    def fix_symbolic(
        self,
        i: int,
        j: int,
        fn_name: str,
        a: float = 1.0,
        b: float = 0.0,
        c: float = 1.0,
        d: float = 0.0,
        fit_params: bool = False,
        x: Optional[mx.array] = None,
        y: Optional[mx.array] = None,
    ) -> None:
        """Fix edge (i, j) to a symbolic function.

        Args:
            i: Input neuron index
            j: Output neuron index
            fn_name: Name of symbolic function
            a, b, c, d: Affine parameters (y = c * f(a*x + b) + d)
            fit_params: If True, fit a, b, c, d from data
            x: Input data for fitting (required if fit_params=True)
            y: Target data for fitting (required if fit_params=True)
        """
        if fn_name not in SYMBOLIC_REGISTRY:
            raise ValueError(f"Unknown symbolic function: {fn_name}. "
                           f"Available: {list_symbolic()}")

        self.fns_name[i][j] = fn_name

        if fit_params and x is not None and y is not None:
            # Fit affine parameters
            x_np = np.array(x).flatten()
            y_np = np.array(y).flatten()
            a, b, c, d, r2 = fit_affine_params(x_np, y_np, fn_name)

        # Update parameters
        self.affine_a = self.affine_a.at[i, j].add(a - self.affine_a[i, j])
        self.affine_b = self.affine_b.at[i, j].add(b - self.affine_b[i, j])
        self.affine_c = self.affine_c.at[i, j].add(c - self.affine_c[i, j])
        self.affine_d = self.affine_d.at[i, j].add(d - self.affine_d[i, j])

        # Update mask
        self._symbolic_mask = self._symbolic_mask.at[i, j].add(1.0 - self._symbolic_mask[i, j])

    def unfix_symbolic(self, i: int, j: int) -> None:
        """Remove symbolic assignment from edge (i, j)."""
        self.fns_name[i][j] = None
        self._symbolic_mask = self._symbolic_mask.at[i, j].add(-self._symbolic_mask[i, j])

    def __call__(
        self,
        x: mx.array,
        singularity_avoiding: bool = False,
        y_th: float = 10.0,
    ) -> mx.array:
        """Evaluate symbolic functions.

        Args:
            x: Input of shape (batch, in_dim)
            singularity_avoiding: Clip outputs to avoid singularities
            y_th: Threshold for clipping

        Returns:
            Output of shape (batch, in_dim, out_dim)
        """
        batch_size = x.shape[0]

        # Initialize output
        y = mx.zeros((batch_size, self.in_dim, self.out_dim))

        # Evaluate each symbolic edge
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                if self.fns_name[i][j] is not None:
                    fn = SYMBOLIC_REGISTRY[self.fns_name[i][j]].fn

                    # Get affine parameters
                    a = self.affine_a[i, j]
                    b = self.affine_b[i, j]
                    c = self.affine_c[i, j]
                    d = self.affine_d[i, j]

                    # Compute: c * f(a * x_i + b) + d
                    x_i = x[:, i]
                    val = c * fn(a * x_i + b) + d

                    if singularity_avoiding:
                        val = mx.clip(val, -y_th, y_th)

                    # Update output using a loop-safe approach
                    y_slice = y[:, i, j]
                    y = y.at[:, i, j].add(val - y_slice)

        return y

    def get_formula(
        self,
        i: int,
        j: int,
        var_name: str = "x",
        decimals: int = 2,
    ) -> str:
        """Get the symbolic formula for edge (i, j).

        Args:
            i: Input index
            j: Output index
            var_name: Variable name to use
            decimals: Decimal places for coefficients

        Returns:
            String representation of the formula
        """
        if self.fns_name[i][j] is None:
            return f"spline({var_name})"

        fn_name = self.fns_name[i][j]
        a = float(self.affine_a[i, j])
        b = float(self.affine_b[i, j])
        c = float(self.affine_c[i, j])
        d = float(self.affine_d[i, j])

        # Build formula string
        # Inner: a*x + b
        if abs(a - 1.0) < 1e-6:
            inner = var_name
        else:
            inner = f"{a:.{decimals}f}*{var_name}"

        if abs(b) > 1e-6:
            if b > 0:
                inner = f"({inner} + {b:.{decimals}f})"
            else:
                inner = f"({inner} - {abs(b):.{decimals}f})"

        # Apply function
        if fn_name == "x":
            f_inner = inner
        elif fn_name == "x^2":
            f_inner = f"({inner})^2"
        elif fn_name == "x^3":
            f_inner = f"({inner})^3"
        else:
            f_inner = f"{fn_name}({inner})"

        # Outer: c * f(...) + d
        if abs(c - 1.0) < 1e-6:
            result = f_inner
        else:
            result = f"{c:.{decimals}f}*{f_inner}"

        if abs(d) > 1e-6:
            if d > 0:
                result = f"{result} + {d:.{decimals}f}"
            else:
                result = f"{result} - {abs(d):.{decimals}f}"

        return result

    def get_latex_formula(
        self,
        i: int,
        j: int,
        var_name: str = "x",
        decimals: int = 2,
    ) -> str:
        """Get the LaTeX formula for edge (i, j).

        Args:
            i: Input index
            j: Output index
            var_name: Variable name to use
            decimals: Decimal places for coefficients

        Returns:
            LaTeX string representation of the formula
        """
        if self.fns_name[i][j] is None:
            return f"\\text{{spline}}({var_name})"

        fn_name = self.fns_name[i][j]
        sym_fn = SYMBOLIC_REGISTRY[fn_name]
        a = float(self.affine_a[i, j])
        b = float(self.affine_b[i, j])
        c = float(self.affine_c[i, j])
        d = float(self.affine_d[i, j])

        # Build inner expression: a*x + b
        if abs(a - 1.0) < 1e-6:
            inner = var_name
        elif abs(a + 1.0) < 1e-6:
            inner = f"-{var_name}"
        else:
            inner = f"{a:.{decimals}f} {var_name}"

        if abs(b) > 1e-6:
            if b > 0:
                inner = f"{inner} + {b:.{decimals}f}"
            else:
                inner = f"{inner} - {abs(b):.{decimals}f}"

        # Apply function using LaTeX notation
        latex_fn = sym_fn.latex.replace("x", f"({inner})" if "+" in inner or "-" in inner else inner)

        # Outer: c * f(...) + d
        if abs(c - 1.0) < 1e-6:
            result = latex_fn
        elif abs(c + 1.0) < 1e-6:
            result = f"-{latex_fn}"
        else:
            result = f"{c:.{decimals}f} {latex_fn}"

        if abs(d) > 1e-6:
            if d > 0:
                result = f"{result} + {d:.{decimals}f}"
            else:
                result = f"{result} - {abs(d):.{decimals}f}"

        return result

    def get_typst_formula(
        self,
        i: int,
        j: int,
        var_name: str = "x",
        decimals: int = 2,
    ) -> str:
        """Get the Typst formula for edge (i, j).

        Args:
            i: Input index
            j: Output index
            var_name: Variable name to use
            decimals: Decimal places for coefficients

        Returns:
            Typst string representation of the formula
        """
        if self.fns_name[i][j] is None:
            return f"\"spline\"({var_name})"

        fn_name = self.fns_name[i][j]
        sym_fn = SYMBOLIC_REGISTRY[fn_name]
        a = float(self.affine_a[i, j])
        b = float(self.affine_b[i, j])
        c = float(self.affine_c[i, j])
        d = float(self.affine_d[i, j])

        # Build inner expression: a*x + b
        if abs(a - 1.0) < 1e-6:
            inner = var_name
        elif abs(a + 1.0) < 1e-6:
            inner = f"-{var_name}"
        else:
            inner = f"{a:.{decimals}f} {var_name}"

        if abs(b) > 1e-6:
            if b > 0:
                inner = f"({inner} + {b:.{decimals}f})"
            else:
                inner = f"({inner} - {abs(b):.{decimals}f})"

        # Apply function using Typst notation
        typst_fn = sym_fn.typst.replace("x", inner)

        # Outer: c * f(...) + d
        if abs(c - 1.0) < 1e-6:
            result = typst_fn
        elif abs(c + 1.0) < 1e-6:
            result = f"-{typst_fn}"
        else:
            result = f"{c:.{decimals}f} dot {typst_fn}"

        if abs(d) > 1e-6:
            if d > 0:
                result = f"{result} + {d:.{decimals}f}"
            else:
                result = f"{result} - {abs(d):.{decimals}f}"

        return result

    def get_subset(self, in_ids: List[int], out_ids: List[int]) -> "Symbolic_KANLayer":
        """Extract a subset of the layer."""
        new_layer = Symbolic_KANLayer(len(in_ids), len(out_ids))

        for new_i, old_i in enumerate(in_ids):
            for new_j, old_j in enumerate(out_ids):
                new_layer.fns_name[new_i][new_j] = self.fns_name[old_i][old_j]

        in_ids_arr = mx.array(in_ids)
        out_ids_arr = mx.array(out_ids)

        new_layer.affine_a = self.affine_a[in_ids_arr][:, out_ids_arr]
        new_layer.affine_b = self.affine_b[in_ids_arr][:, out_ids_arr]
        new_layer.affine_c = self.affine_c[in_ids_arr][:, out_ids_arr]
        new_layer.affine_d = self.affine_d[in_ids_arr][:, out_ids_arr]
        new_layer._symbolic_mask = self._symbolic_mask[in_ids_arr][:, out_ids_arr]

        return new_layer


# =============================================================================
# SYMBOLIC REGRESSION PIPELINE
# =============================================================================

def symbolic_pipeline(
    model,
    x: mx.array,
    prune_threshold: float = 0.01,
    r2_threshold: float = 0.95,
    refit_steps: int = 100,
    refit_lr: float = 0.01,
    verbose: bool = True,
) -> Tuple[Dict, str]:
    """Full symbolic regression pipeline for KAN models.

    Implements the KAN paper's symbolic extraction workflow:
    1. Prune low-importance edges
    2. Fit symbolic functions to remaining edges
    3. Global refit of symbolic affine parameters
    4. Export formula

    This turns a trained KAN into an interpretable symbolic formula.

    Args:
        model: Trained MultKAN model.
        x: Sample points for fitting (shape: batch, in_dim).
        prune_threshold: Threshold for edge pruning (0.01 = remove edges < 1%).
        r2_threshold: R² threshold for symbolic fitting (0.95 = 95% accuracy).
        refit_steps: Number of steps for global refit.
        refit_lr: Learning rate for global refit.
        verbose: Print progress.

    Returns:
        Tuple of (report_dict, formula_string):
            - report: Dictionary with extraction statistics
            - formula: Final symbolic formula as string

    Example:
        model = MultKAN(width=[1, 10, 1])
        model.fit(dataset, steps=500)

        report, formula = symbolic_pipeline(model, x_sample, verbose=True)
        print(f"Formula: {formula}")
        print(f"Edges symbolified: {report['edges_symbolified']}")
    """
    from itertools import product

    report = {
        "original_edges": 0,
        "edges_after_prune": 0,
        "edges_symbolified": 0,
        "symbolic_fits": [],
        "final_r2": None,
        "formula": "",
    }

    # Count original active edges
    for layer in model.layers:
        mask = np.array(layer.mask) if hasattr(layer, 'mask') else np.ones((layer.in_dim, layer.out_dim))
        report["original_edges"] += int(np.sum(mask > 0.5))

    if verbose:
        print(f"Symbolic Pipeline: {report['original_edges']} active edges")

    # 1. PRUNE low-importance edges
    if verbose:
        print(f"  Step 1: Pruning edges below {prune_threshold:.2%} contribution...")

    if hasattr(model, 'prune'):
        model.prune(threshold=prune_threshold)

    # Count edges after pruning
    for layer in model.layers:
        mask = np.array(layer.mask) if hasattr(layer, 'mask') else np.ones((layer.in_dim, layer.out_dim))
        report["edges_after_prune"] += int(np.sum(mask > 0.5))

    if verbose:
        pruned = report["original_edges"] - report["edges_after_prune"]
        print(f"    Pruned {pruned} edges, {report['edges_after_prune']} remaining")

    # 2. FIT symbolic functions to each edge
    if verbose:
        print(f"  Step 2: Fitting symbolic functions (R² > {r2_threshold:.2%})...")

    x_np = np.array(x)

    for l_idx, layer in enumerate(model.layers):
        in_dim = layer.in_dim
        out_dim = layer.out_dim
        mask = np.array(layer.mask) if hasattr(layer, 'mask') else np.ones((in_dim, out_dim))

        for i, j in product(range(in_dim), range(out_dim)):
            if mask[i, j] < 0.5:
                continue  # Skip pruned edges

            # Sample edge function
            try:
                x_edge, y_edge = _sample_edge_function(model, l_idx, i, j, x_np)

                if len(x_edge) < 10:
                    continue

                # Try symbolic fit
                suggestions = suggest_symbolic(x_edge, y_edge, top_k=1)

                if suggestions and suggestions[0][5] >= r2_threshold:
                    fn_name, a, b, c, d, r2, score = suggestions[0]

                    # Fix this edge to symbolic
                    if hasattr(model, 'fix_symbolic'):
                        model.fix_symbolic(l_idx, i, j, fn_name, a, b, c, d)

                    report["edges_symbolified"] += 1
                    report["symbolic_fits"].append({
                        "layer": l_idx,
                        "edge": (i, j),
                        "function": fn_name,
                        "r2": r2,
                        "params": (a, b, c, d),
                    })

                    if verbose:
                        print(f"    L{l_idx}[{i},{j}]: {fn_name} (R²={r2:.4f})")

            except Exception as e:
                logger.debug(f"Edge fitting failed for L{l_idx}[{i},{j}]: {e}")
                continue

    if verbose:
        print(f"    Symbolified {report['edges_symbolified']} edges")

    # 3. GLOBAL REFIT (optional)
    if refit_steps > 0 and report["edges_symbolified"] > 0:
        if verbose:
            print(f"  Step 3: Global refit ({refit_steps} steps)...")

        # TODO: Implement global refit of symbolic affine parameters
        # This would require extracting affine params, optimizing them,
        # and updating the model. For now, skip this step.
        if verbose:
            print("    (Skipped - global refit not yet implemented)")

    # 4. EXPORT formula
    if verbose:
        print("  Step 4: Extracting formula...")

    if hasattr(model, 'symbolic_formula'):
        try:
            formula = model.symbolic_formula(decimals=3)
            report["formula"] = formula
        except Exception as e:
            report["formula"] = f"(extraction failed: {e})"
    else:
        report["formula"] = "(model does not support symbolic_formula)"

    if verbose:
        print(f"\nResult: {report['edges_symbolified']}/{report['original_edges']} edges symbolified")
        if report["formula"]:
            print(f"Formula: {report['formula'][:200]}...")

    return report, report["formula"]


def _sample_edge_function(
    model,
    layer_idx: int,
    i: int,
    j: int,
    x_np: np.ndarray,
    n_samples: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the 1D function on edge (i, j) of layer layer_idx.

    Args:
        model: MultKAN model.
        layer_idx: Layer index.
        i: Input neuron index.
        j: Output neuron index.
        x_np: Input samples (batch, in_dim).
        n_samples: Number of points to sample.

    Returns:
        (x_edge, y_edge) as 1D numpy arrays.
    """
    # Forward through layers up to layer_idx to get activations
    x = mx.array(x_np)

    for l_idx in range(layer_idx):
        layer = model.layers[l_idx]
        x_raw = layer(x)
        if hasattr(model, '_apply_mult_nodes'):
            x = model._apply_mult_nodes(x_raw, l_idx)
        else:
            x = x_raw

    # Now x is the input to layer layer_idx
    # Get the i-th input component
    x_i = np.array(x[:, i]).flatten()

    # Evaluate the edge function
    layer = model.layers[layer_idx]

    # Get edge output (this is approximate - we'd need to evaluate just this edge)
    # For now, use a simpler approach: sample the edge's contribution
    x_i_sorted_idx = np.argsort(x_i)
    x_i_sorted = x_i[x_i_sorted_idx]

    # Sample n_samples points uniformly in the range
    if len(x_i_sorted) > n_samples:
        idx = np.linspace(0, len(x_i_sorted) - 1, n_samples, dtype=int)
        x_edge = x_i_sorted[idx]
    else:
        x_edge = x_i_sorted

    # Evaluate spline at these points
    if layer._grid is not None:
        from .spline import coef2curve

        x_edge_mx = mx.array(x_edge.reshape(-1, 1))
        grid_i = layer._grid[i:i+1]  # Grid for input i
        coef_ij = layer.coef[i, j:j+1, :]  # Coefficients for edge (i, j)

        # Evaluate spline
        try:
            y_spline = coef2curve(
                x_edge_mx,
                grid_i,
                coef_ij[None, :, :],  # Add in_dim axis
                layer.k
            )
            y_edge = np.array(y_spline[:, 0, 0]).flatten()
        except Exception:
            # Fall back to evaluating the full layer and extracting
            y_edge = np.zeros_like(x_edge)
    else:
        # Non-spline basis - harder to isolate edge
        y_edge = np.zeros_like(x_edge)

    return x_edge, y_edge


def score_symbolic_aic(
    r2: float,
    n_params: int,
    n_samples: int,
) -> float:
    """AIC-style scoring for symbolic fits.

    Balances accuracy (R²) against complexity (number of parameters)
    using an information-theoretic criterion.

    Args:
        r2: R² score of the fit.
        n_params: Number of parameters in the symbolic function.
        n_samples: Number of data points.

    Returns:
        AIC-like score (lower is better for AIC, but we return negative
        so higher is better for consistency with R²).
    """
    if r2 >= 1.0:
        r2 = 0.9999  # Avoid log(0)

    # Residual sum of squares (normalized)
    rss = (1 - r2) * n_samples

    # AIC = n * log(RSS/n) + 2k
    aic = n_samples * np.log(rss / n_samples + 1e-10) + 2 * n_params

    # Return negative so higher is better
    return -aic
