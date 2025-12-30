"""Physics-Informed Neural Network (PINN) utilities for KAN-MLX-Physics.

This module provides derivative operators and helpers for solving PDEs
using the batch-grad sum trick for efficient automatic differentiation.

The key insight is that mx.grad requires scalar outputs, so we differentiate
the SUM over the batch to get per-sample gradients:

    d(Σᵢ uᵢ)/dxⱼ = duⱼ/dxⱼ  (for batch-separable functions)

Example:
    from kan_mlx_physics import MultKAN
    from kan_mlx_physics.pinn import make_derivative_fns, make_laplacian_fn

    model = MultKAN(width=[1, 2, 1], grid=5, k=3)
    u_fn, du_dx_fn, d2u_dx2_fn = make_derivative_fns(model)

    # Use in loss function
    def pde_loss(params, x):
        u = u_fn(params, x)
        u_xx = d2u_dx2_fn(params, x)
        residual = u_xx + u  # Example: u'' + u = 0
        return mx.mean(residual**2)
"""

import mlx.core as mx
from typing import Callable, Tuple, Optional

from .functional import functional_forward, ParamsList


def make_u_fn(
    k: int,
    base_fun: Callable[[mx.array], mx.array],
) -> Callable[[ParamsList, mx.array], mx.array]:
    """Create a function that computes u(x) for a batch of points.

    Args:
        k: Spline order
        base_fun: Base activation function

    Returns:
        Function u_fn(params, x) -> (N,) array of function values
    """
    def u_fn(params_list: ParamsList, x_batch: mx.array) -> mx.array:
        """Compute u(x) for batch. Returns shape (N,)."""
        y = functional_forward(params_list, x_batch, k, base_fun)  # (N, out_dim)
        return mx.squeeze(y, axis=-1)  # (N,)

    return u_fn


def make_derivative_fns(
    k: int,
    base_fun: Callable[[mx.array], mx.array],
) -> Tuple[
    Callable[[ParamsList, mx.array], mx.array],
    Callable[[ParamsList, mx.array], mx.array],
    Callable[[ParamsList, mx.array], mx.array],
]:
    """Create functions for u, du/dx, and d²u/dx² using the batch-grad sum trick.

    This is the recommended approach for 1D PDEs. It avoids vmap overhead
    by differentiating the sum over the batch.

    Args:
        k: Spline order
        base_fun: Base activation function

    Returns:
        Tuple of (u_fn, du_dx_fn, d2u_dx2_fn) where each takes (params, x)

    Example:
        u_fn, du_dx, d2u_dx2 = make_derivative_fns(model.k, model.base_fun)

        def schrodinger_loss(params, x, E):
            psi = u_fn(params, x)
            psi_xx = d2u_dx2(params, x)
            residual = -0.5 * psi_xx - E * psi
            return mx.mean(residual**2)
    """
    # u(params, x) -> (N,)
    def u_vec(params_list: ParamsList, x_batch: mx.array) -> mx.array:
        y = functional_forward(params_list, x_batch, k, base_fun)
        return mx.squeeze(y, axis=-1)

    # Sum over batch -> scalar (required by mx.grad)
    def u_sum(params_list: ParamsList, x_batch: mx.array) -> mx.array:
        return mx.sum(u_vec(params_list, x_batch))

    # First derivative: d(sum u)/dx gives per-sample du/dx
    _du_dx_raw = mx.grad(u_sum, argnums=1)

    def du_dx(params_list: ParamsList, x_batch: mx.array) -> mx.array:
        """Compute du/dx for batch. Returns shape (N,) or (N, in_dim)."""
        grad = _du_dx_raw(params_list, x_batch)  # (N, in_dim)
        if grad.shape[-1] == 1:
            return mx.squeeze(grad, axis=-1)  # (N,) for 1D
        return grad  # (N, in_dim) for multi-D

    # Sum of first derivatives -> scalar
    def du_sum(params_list: ParamsList, x_batch: mx.array) -> mx.array:
        return mx.sum(_du_dx_raw(params_list, x_batch))

    # Second derivative: d(sum du/dx)/dx gives per-sample d²u/dx²
    _d2u_dx2_raw = mx.grad(du_sum, argnums=1)

    def d2u_dx2(params_list: ParamsList, x_batch: mx.array) -> mx.array:
        """Compute d²u/dx² for batch. Returns shape (N,) or (N, in_dim)."""
        grad = _d2u_dx2_raw(params_list, x_batch)  # (N, in_dim)
        if grad.shape[-1] == 1:
            return mx.squeeze(grad, axis=-1)  # (N,) for 1D
        return grad  # (N, in_dim) for multi-D

    return u_vec, du_dx, d2u_dx2


def make_laplacian_fn(
    k: int,
    base_fun: Callable[[mx.array], mx.array],
    dim: int = 1,
) -> Callable[[ParamsList, mx.array], mx.array]:
    """Create a Laplacian function (∇²u) for arbitrary dimensions.

    For 1D: ∇²u = d²u/dx²
    For 2D: ∇²u = d²u/dx² + d²u/dy²
    For 3D: ∇²u = d²u/dx² + d²u/dy² + d²u/dz²

    Args:
        k: Spline order
        base_fun: Base activation function
        dim: Spatial dimension (1, 2, or 3)

    Returns:
        Function laplacian_fn(params, x) -> (N,) array
    """
    _, _, d2u_dx2 = make_derivative_fns(k, base_fun)

    if dim == 1:
        return d2u_dx2

    def laplacian(params_list: ParamsList, x_batch: mx.array) -> mx.array:
        """Compute ∇²u = Σᵢ d²u/dxᵢ². Returns shape (N,)."""
        d2u = d2u_dx2(params_list, x_batch)  # (N, dim)
        return mx.sum(d2u, axis=-1)  # (N,)

    return laplacian


def make_gradient_fn(
    k: int,
    base_fun: Callable[[mx.array], mx.array],
) -> Callable[[ParamsList, mx.array], mx.array]:
    """Create a gradient function (∇u) for arbitrary dimensions.

    Args:
        k: Spline order
        base_fun: Base activation function

    Returns:
        Function gradient_fn(params, x) -> (N, dim) array
    """
    _, du_dx, _ = make_derivative_fns(k, base_fun)
    return du_dx


def finite_difference_laplacian(
    forward_fn: Callable[[mx.array], mx.array],
    x: mx.array,
    h: float = 1e-3,
) -> Tuple[mx.array, mx.array]:
    """Compute Laplacian using central finite differences.

    This is faster but less accurate than autograd. Useful for quick
    experiments or when autograd has issues.

    d²u/dx² ≈ (u(x+h) - 2u(x) + u(x-h)) / h²

    Args:
        forward_fn: Function that takes x and returns u(x)
        x: Input tensor of shape (N, dim)
        h: Step size for finite differences

    Returns:
        Tuple of (u, laplacian_u) each of shape (N, out_dim)
    """
    u_center = forward_fn(x)
    u_plus = forward_fn(x + h)
    u_minus = forward_fn(x - h)
    laplacian = (u_plus - 2 * u_center + u_minus) / (h * h)
    return u_center, laplacian


class PINNOperators:
    """Container for PINN differential operators.

    This class pre-builds derivative functions for a given model configuration,
    making it easy to use in loss functions.

    Example:
        ops = PINNOperators(model.k, model.base_fun)

        def loss(params, x, E):
            psi = ops.u(params, x)
            psi_xx = ops.laplacian(params, x)
            residual = -0.5 * psi_xx - E * psi
            return mx.mean(residual**2)
    """

    def __init__(
        self,
        k: int,
        base_fun: Callable[[mx.array], mx.array],
        dim: int = 1,
    ):
        """Initialize PINN operators.

        Args:
            k: Spline order
            base_fun: Base activation function
            dim: Spatial dimension
        """
        self.k = k
        self.base_fun = base_fun
        self.dim = dim

        # Build derivative functions
        self._u, self._du_dx, self._d2u_dx2 = make_derivative_fns(k, base_fun)

    def u(self, params: ParamsList, x: mx.array) -> mx.array:
        """Compute u(x). Returns shape (N,)."""
        return self._u(params, x)

    def gradient(self, params: ParamsList, x: mx.array) -> mx.array:
        """Compute ∇u. Returns shape (N,) for 1D or (N, dim) for multi-D."""
        return self._du_dx(params, x)

    def laplacian(self, params: ParamsList, x: mx.array) -> mx.array:
        """Compute ∇²u. Returns shape (N,)."""
        if self.dim == 1:
            return self._d2u_dx2(params, x)
        else:
            d2u = self._d2u_dx2(params, x)  # (N, dim)
            return mx.sum(d2u, axis=-1)  # (N,)

    def dx(self, params: ParamsList, x: mx.array, component: int = 0) -> mx.array:
        """Compute du/dxᵢ for a specific component. Returns shape (N,)."""
        grad = self._du_dx(params, x)
        if self.dim == 1:
            return grad
        return grad[:, component]

    def d2x(self, params: ParamsList, x: mx.array, component: int = 0) -> mx.array:
        """Compute d²u/dxᵢ² for a specific component. Returns shape (N,)."""
        d2u = self._d2u_dx2(params, x)
        if self.dim == 1:
            return d2u
        return d2u[:, component]


def make_compiled_pinn_step(
    k: int,
    base_fun: Callable[[mx.array], mx.array],
    loss_fn: Callable,
    lr: float = 0.003,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    """Create a compiled training step for PINN optimization.

    This returns a @mx.compile decorated function that performs one
    optimization step, including loss computation, gradient calculation,
    and Adam update.

    Args:
        k: Spline order
        base_fun: Base activation function
        loss_fn: Loss function with signature (params, ...) -> scalar
        lr: Learning rate
        beta1: Adam beta1
        beta2: Adam beta2
        eps: Adam epsilon

    Returns:
        Compiled function: train_step(params, m, v, t, *loss_args) -> (loss, params, m, v)

    Example:
        def my_loss(params, x, E):
            u = u_fn(params, x)
            u_xx = d2u_dx2_fn(params, x)
            return mx.mean((u_xx + E*u)**2)

        train_step = make_compiled_pinn_step(model.k, model.base_fun, my_loss)

        for epoch in range(1000):
            loss, params, m, v = train_step(params, m, v, t, x_train, E)
    """
    from .functional import adam_update

    loss_and_grad = mx.value_and_grad(loss_fn)

    @mx.compile
    def train_step(params, m_params, v_params, t, *loss_args):
        loss, grads = loss_and_grad(params, *loss_args)
        new_params, new_m, new_v = adam_update(
            params, grads, m_params, v_params, t, lr, beta1, beta2, eps
        )
        return loss, new_params, new_m, new_v

    return train_step
