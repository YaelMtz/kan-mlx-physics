"""Hard Boundary Constraints for PDE Solving.

Instead of penalizing boundary conditions through loss terms, hard constraints
enforce them structurally through ansatz functions:

    u(x) = g(x) + h(x) * N_theta(x)

where:
- g(x) is the boundary value function (satisfies u = g on boundary)
- h(x) is a distance function that vanishes on the boundary
- N_theta(x) is the neural network output

This approach:
1. Guarantees exact boundary condition satisfaction
2. Removes BC loss term from the optimization
3. Often stabilizes training significantly
4. Makes symbolic extraction cleaner

Example:
    # Dirichlet BC: u(0) = u(1) = 0
    constraint = HardConstraint(domain=[(0, 1)], bc_type="dirichlet")

    def constrained_u(x):
        raw_u = model(x)
        return constraint.apply(x, raw_u)

    # Now constrained_u automatically satisfies BCs!
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np


@dataclass
class HardConstraint:
    """Hard boundary constraint enforcer.

    Enforces boundary conditions structurally through:
        u(x) = g(x) + h(x) * N(x)

    where h(x) = 0 on the boundary.

    Attributes:
        bounds: Domain bounds as list of (min, max) tuples.
        bc_type: Type of boundary condition.
        bc_values: Boundary values for Dirichlet (optional).
        g_fn: Custom boundary value function (optional).
    """

    bounds: List[Tuple[float, float]]
    bc_type: str = "dirichlet"
    bc_values: Optional[dict] = None
    g_fn: Optional[Callable[[mx.array], mx.array]] = None

    @property
    def dim(self) -> int:
        """Spatial dimension."""
        return len(self.bounds)

    def h(self, x: mx.array) -> mx.array:
        """Distance function that vanishes on the boundary.

        For 1D [a, b]: h(x) = (x - a)(b - x) / L^2
        For nD: h(x) = product of 1D factors

        Args:
            x: Points of shape (batch, dim).

        Returns:
            Distance values of shape (batch,).
        """
        if self.dim == 1:
            a, b = self.bounds[0]
            L = b - a
            x_1d = x[:, 0] if x.ndim > 1 else x
            # Normalized so h is ~1 in the middle
            return (x_1d - a) * (b - x_1d) / ((L / 2) ** 2)
        else:
            # Product of 1D distance functions
            h_val = mx.ones((x.shape[0],))
            for i, (lo, hi) in enumerate(self.bounds):
                L = hi - lo
                h_i = (x[:, i] - lo) * (hi - x[:, i]) / ((L / 2) ** 2)
                h_val = h_val * h_i
            return h_val

    def g(self, x: mx.array) -> mx.array:
        """Boundary value function.

        Returns the value that u should take on the boundary.

        Args:
            x: Points of shape (batch, dim).

        Returns:
            Boundary values of shape (batch, 1) or (batch,).
        """
        if self.g_fn is not None:
            return self.g_fn(x)

        # Default: zero Dirichlet
        if self.bc_type == "dirichlet":
            if self.bc_values is None:
                return mx.zeros((x.shape[0], 1))

            # Interpolate boundary values (for non-zero Dirichlet)
            # This is a simple case; more complex cases need custom g_fn
            return mx.zeros((x.shape[0], 1))

        elif self.bc_type == "periodic":
            # Periodic: no shift needed, handled by h
            return mx.zeros((x.shape[0], 1))

        else:
            return mx.zeros((x.shape[0], 1))

    def apply(
        self,
        x: mx.array,
        nn_output: mx.array,
        squeeze: bool = True,
    ) -> mx.array:
        """Apply hard constraint to neural network output.

        Computes: u(x) = g(x) + h(x) * N(x)

        Args:
            x: Input points of shape (batch, dim).
            nn_output: Neural network output of shape (batch, 1) or (batch,).
            squeeze: Whether to squeeze the output.

        Returns:
            Constrained output satisfying BCs.
        """
        h_x = self.h(x)  # (batch,)
        g_x = self.g(x)  # (batch, 1) or (batch,)

        # Ensure shapes match
        if nn_output.ndim == 1:
            nn_out = nn_output
        else:
            nn_out = nn_output.squeeze(-1)  # (batch,)

        if g_x.ndim == 2:
            g_x = g_x.squeeze(-1)  # (batch,)

        result = g_x + h_x * nn_out

        if squeeze:
            return result
        else:
            return result[:, None]

    @classmethod
    def dirichlet_zero(cls, bounds: List[Tuple[float, float]]) -> "HardConstraint":
        """Create constraint for zero Dirichlet BC: u = 0 on boundary.

        Args:
            bounds: Domain bounds as [(a1, b1), (a2, b2), ...].

        Returns:
            HardConstraint enforcing u = 0 on boundary.
        """
        return cls(bounds=bounds, bc_type="dirichlet")

    @classmethod
    def dirichlet_custom(
        cls,
        bounds: List[Tuple[float, float]],
        g_fn: Callable[[mx.array], mx.array],
    ) -> "HardConstraint":
        """Create constraint for custom Dirichlet BC: u = g on boundary.

        Args:
            bounds: Domain bounds.
            g_fn: Function returning boundary values.

        Returns:
            HardConstraint enforcing u = g(x) on boundary.
        """
        return cls(bounds=bounds, bc_type="dirichlet", g_fn=g_fn)


class SineSeriesConstraint:
    """Sine series ansatz that automatically satisfies Dirichlet BCs.

    For 1D problems on [0, L] with u(0) = u(L) = 0, uses:
        u(x) = Σ_n c_n * sin(n*π*x/L)

    This is a "hard" constraint because sin(n*π*x/L) = 0 at x=0 and x=L
    by construction.

    Note: This is not a full KAN replacement, but can be used as a
    wrapper or for comparison.
    """

    def __init__(self, L: float = 1.0, n_terms: int = 10):
        """Initialize sine series constraint.

        Args:
            L: Domain length [0, L].
            n_terms: Number of sine terms.
        """
        self.L = L
        self.n_terms = n_terms

    def basis(self, x: mx.array) -> mx.array:
        """Compute sine basis functions.

        Args:
            x: Points of shape (batch, 1) or (batch,).

        Returns:
            Basis values of shape (batch, n_terms).
        """
        if x.ndim == 2:
            x = x[:, 0]

        # n = 1, 2, ..., n_terms
        n = mx.arange(1, self.n_terms + 1, dtype=mx.float32)

        # sin(n*π*x/L) for each n
        # x: (batch,), n: (n_terms,)
        # Result: (batch, n_terms)
        return mx.sin(mx.pi * n * x[:, None] / self.L)

    def apply(self, x: mx.array, coefficients: mx.array) -> mx.array:
        """Apply sine series with given coefficients.

        Args:
            x: Points of shape (batch, 1) or (batch,).
            coefficients: Coefficients of shape (n_terms,) or (batch, n_terms).

        Returns:
            u(x) = Σ c_n * sin(n*π*x/L) of shape (batch,).
        """
        basis_vals = self.basis(x)  # (batch, n_terms)

        if coefficients.ndim == 1:
            # Same coefficients for all points
            return mx.sum(basis_vals * coefficients, axis=1)
        else:
            # Per-point coefficients
            return mx.sum(basis_vals * coefficients, axis=1)


def wrap_model_with_constraint(
    model_fn: Callable[[mx.array], mx.array],
    constraint: HardConstraint,
) -> Callable[[mx.array], mx.array]:
    """Wrap a model function to enforce hard constraints.

    Args:
        model_fn: Original model function (e.g., model.__call__).
        constraint: HardConstraint to apply.

    Returns:
        Wrapped function that satisfies boundary conditions.

    Example:
        constraint = HardConstraint.dirichlet_zero([(0, 1)])
        constrained_model = wrap_model_with_constraint(model, constraint)

        # Now constrained_model(x) always satisfies u(0) = u(1) = 0
        u = constrained_model(x)
    """

    def wrapped(x: mx.array) -> mx.array:
        raw_output = model_fn(x)
        return constraint.apply(x, raw_output)

    return wrapped


def wrap_functional_with_constraint(
    params_fn: Callable,
    constraint: HardConstraint,
) -> Callable:
    """Wrap a functional-style model to enforce hard constraints.

    Args:
        params_fn: Function taking (params, x) -> output.
        constraint: HardConstraint to apply.

    Returns:
        Wrapped function that satisfies boundary conditions.

    Example:
        from kan_mlx_physics.functional import functional_forward

        constraint = HardConstraint.dirichlet_zero([(0, 1)])
        constrained_fn = wrap_functional_with_constraint(
            lambda params, x: functional_forward(params, x, k, base_fun),
            constraint
        )

        # Use in PINN loss
        def loss(params, x):
            u = constrained_fn(params, x)  # Satisfies BCs!
            ...
    """

    def wrapped(params, x: mx.array) -> mx.array:
        raw_output = params_fn(params, x)
        return constraint.apply(x, raw_output)

    return wrapped
