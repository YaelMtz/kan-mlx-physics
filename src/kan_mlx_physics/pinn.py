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

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING, Optional, List, Dict, Any

import mlx.core as mx

from .functional import ParamsList, functional_forward


class CompileMode(Enum):
    """Compilation mode for training.

    MLX's mx.compile accelerates computation but not all operations have VJP
    (vector-Jacobian product) rules for backpropagation. This enum controls
    how compilation is applied to avoid "Cannot vjp primitive" errors.

    Modes:
        EAGER: No compilation. Safest, always works, slower.
        FORWARD_ONLY: Compile only forward pass. Recommended for most cases.
            The forward pass runs on Metal, gradients use standard rules.
        FULL: Compile entire loss_and_grad. Fastest but may fail with
            unsupported operations (argmax, sort, some reductions).
        AUTO: Try FULL, fallback to FORWARD_ONLY, then EAGER on failure.
    """
    EAGER = "eager"
    FORWARD_ONLY = "forward_only"
    FULL = "full"
    AUTO = "auto"

if TYPE_CHECKING:
    from .multkan import MultKAN


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
) -> tuple[
    Callable[[ParamsList, mx.array], mx.array],
    Callable[[ParamsList, mx.array], mx.array],
    Callable[[ParamsList, mx.array], mx.array],
]:
    """Create functions for u, du/dx, and d2u/dx2 using the batch-grad sum trick.

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


def make_model_derivative_fns(
    model: "MultKAN",
) -> tuple[
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
]:
    """Create derivative functions using the model directly.

    This works with ANY basis type (B-spline, Laguerre, Hermite, Fourier, etc.)
    by using the model's forward pass instead of the functional API.

    Note: These functions do NOT take params as first arg - they use model params directly.
    Use these for simple training where you don't need custom param handling.

    Args:
        model: MultKAN model instance (any basis type)

    Returns:
        Tuple of (u_fn, du_dx_fn, d2u_dx2_fn) where each takes only x

    Example:
        model = MultKAN(width=[1, 2, 1], basis="laguerre")
        u_fn, du_dx, d2u_dx2 = make_model_derivative_fns(model)

        def loss(x):
            u = u_fn(x)
            u_xx = d2u_dx2(x)
            return mx.mean((u_xx + u)**2)
    """

    def u_vec(x_batch: mx.array) -> mx.array:
        y = model(x_batch)  # (N, out_dim)
        return mx.squeeze(y, axis=-1)  # (N,)

    # Sum over batch -> scalar (required by mx.grad)
    def u_sum(x_batch: mx.array) -> mx.array:
        return mx.sum(u_vec(x_batch))

    # First derivative: d(sum u)/dx gives per-sample du/dx
    _du_dx_raw = mx.grad(u_sum)

    def du_dx(x_batch: mx.array) -> mx.array:
        """Compute du/dx for batch. Returns shape (N,) or (N, in_dim)."""
        grad = _du_dx_raw(x_batch)  # (N, in_dim)
        if grad.shape[-1] == 1:
            return mx.squeeze(grad, axis=-1)  # (N,) for 1D
        return grad  # (N, in_dim) for multi-D

    # Sum of first derivatives -> scalar
    def du_sum(x_batch: mx.array) -> mx.array:
        return mx.sum(_du_dx_raw(x_batch))

    # Second derivative: d(sum du/dx)/dx gives per-sample d²u/dx²
    _d2u_dx2_raw = mx.grad(du_sum)

    def d2u_dx2(x_batch: mx.array) -> mx.array:
        """Compute d²u/dx² for batch. Returns shape (N,) or (N, in_dim)."""
        grad = _d2u_dx2_raw(x_batch)  # (N, in_dim)
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
) -> tuple[mx.array, mx.array]:
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


def make_finite_diff_derivative_fns(
    model: "MultKAN",
    h: float = 1e-3,
) -> tuple[
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
    Callable[[mx.array], mx.array],
]:
    """Create derivative functions using finite differences.

    This is an alternative to autodiff that avoids nested gradient issues.
    Works with ANY basis type and avoids the triple-nested autodiff limitation.

    Uses central differences:
        du/dx ≈ (u(x+h) - u(x-h)) / (2h)
        d²u/dx² ≈ (u(x+h) - 2u(x) + u(x-h)) / h²

    Args:
        model: MultKAN model instance (any basis type)
        h: Step size for finite differences (default 1e-3)

    Returns:
        Tuple of (u_fn, du_dx_fn, d2u_dx2_fn) where each takes only x

    Example:
        model = MultKAN(width=[1, 10, 1], basis="laguerre")
        u_fn, du_dx, d2u_dx2 = make_finite_diff_derivative_fns(model, h=1e-4)

        # Use with nn.value_and_grad for parameter training
        def loss_fn(model, x):
            u = u_fn(x)
            u_xx = d2u_dx2(x)
            return mx.mean((u_xx + u)**2)
    """

    def u_fn(x: mx.array) -> mx.array:
        """Compute u(x). Returns shape (N,)."""
        y = model(x)
        return mx.squeeze(y, axis=-1)

    def du_dx(x: mx.array) -> mx.array:
        """Compute du/dx using central differences. Returns shape (N,) or (N, dim)."""
        in_dim = x.shape[-1]
        if in_dim == 1:
            # 1D: simple central difference
            u_plus = model(x + h)
            u_minus = model(x - h)
            grad = (u_plus - u_minus) / (2.0 * h)
            return mx.squeeze(grad, axis=-1)
        else:
            # Multi-D: compute partial derivatives for each dimension
            # Pre-compute unit vectors scaled by h
            e_vecs = h * mx.eye(in_dim)  # (in_dim, in_dim)
            grads = []
            for i in range(in_dim):
                e_i = e_vecs[i:i+1, :]  # Shape: (1, in_dim)
                u_plus = model(x + e_i)
                u_minus = model(x - e_i)
                grad_i = (u_plus - u_minus) / (2.0 * h)
                grads.append(mx.squeeze(grad_i, axis=-1))
            return mx.stack(grads, axis=-1)

    def d2u_dx2(x: mx.array) -> mx.array:
        """Compute d²u/dx² using central differences. Returns shape (N,) or (N, dim)."""
        in_dim = x.shape[-1]
        u_center = model(x)
        if in_dim == 1:
            # 1D: simple second derivative
            u_plus = model(x + h)
            u_minus = model(x - h)
            d2u = (u_plus - 2.0 * u_center + u_minus) / (h * h)
            return mx.squeeze(d2u, axis=-1)
        else:
            # Multi-D: compute second partial derivatives for each dimension
            # Pre-compute unit vectors scaled by h
            e_vecs = h * mx.eye(in_dim)  # (in_dim, in_dim)
            d2us = []
            for i in range(in_dim):
                e_i = e_vecs[i:i+1, :]  # Shape: (1, in_dim)
                u_plus = model(x + e_i)
                u_minus = model(x - e_i)
                d2u_i = (u_plus - 2.0 * u_center + u_minus) / (h * h)
                d2us.append(mx.squeeze(d2u_i, axis=-1))
            return mx.stack(d2us, axis=-1)

    return u_fn, du_dx, d2u_dx2


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


class PINNTrainer:
    """High-level trainer for Physics-Informed Neural Networks.

    Handles all the complexity of using the functional API with PINNs:
    - Automatic parameter extraction and management
    - Built-in optimizer (Adam, AdamW, SGD)
    - Efficient derivative computation via autodiff or finite differences
    - Automatic model synchronization after training
    - Safe compilation modes to avoid VJP failures

    Supports two derivative methods:
    - "autodiff": Uses batch-grad sum trick (fast, accurate, but may hit
      triple-nested gradient limitations with complex PDE losses)
    - "finite_diff": Uses central finite differences (avoids nested gradient
      issues, works with any basis, slightly less accurate)

    Compile modes (via compile_mode parameter):
    - CompileMode.EAGER: No compilation, always works
    - CompileMode.FORWARD_ONLY: Compile forward pass only (recommended)
    - CompileMode.FULL: Compile entire loss+grad (fastest, may fail)
    - CompileMode.AUTO: Try FULL, fallback to FORWARD_ONLY, then EAGER

    Example (autodiff - default):
        trainer = PINNTrainer(model, lr=0.01)

        def pde_loss(params, x):
            u = trainer.u(params, x)
            u_xx = trainer.d2u(params, x)
            return mx.mean((u_xx + u)**2)

        for epoch in range(1000):
            loss = trainer.step(pde_loss, x_train)

    Example (finite differences - for complex losses):
        trainer = PINNTrainer(model, lr=0.01, derivative_method="finite_diff")

        def pde_loss(model, x):
            u = trainer.u(x)       # No params arg!
            u_xx = trainer.d2u(x)
            return mx.mean((u_xx + u)**2)

        for epoch in range(1000):
            loss = trainer.step(pde_loss, x_train)

    Example (safe compile mode):
        # Recommended: compile forward only to avoid VJP issues
        trainer = PINNTrainer(model, lr=0.01, compile_mode=CompileMode.FORWARD_ONLY)

        # Or auto-fallback on failure
        trainer = PINNTrainer(model, lr=0.01, compile_mode=CompileMode.AUTO)
    """

    def __init__(
        self,
        model: "MultKAN",
        lr: float = 0.001,
        optimizer: str = "adam",
        derivative_method: str = "autodiff",
        finite_diff_h: float = 1e-3,
        compile: bool = False,
        compile_mode: Optional[CompileMode] = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0.0,
    ):
        """Initialize the PINN trainer.

        Args:
            model: MultKAN model instance
            lr: Learning rate
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            derivative_method: 'autodiff' (default) or 'finite_diff'
            finite_diff_h: Step size for finite differences (default 1e-3)
            compile: DEPRECATED. Use compile_mode instead. If True, equivalent
                to compile_mode=CompileMode.FULL.
            compile_mode: Compilation strategy. See CompileMode enum.
                - EAGER: No compilation (safest)
                - FORWARD_ONLY: Compile forward pass only (recommended)
                - FULL: Compile everything (fastest, may fail)
                - AUTO: Try FULL, fallback automatically
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
            weight_decay: Weight decay for AdamW
        """
        import mlx.nn as nn
        import mlx.optimizers as optim

        self.model = model
        self.derivative_method = derivative_method
        self.finite_diff_h = finite_diff_h
        self._lr = lr

        # Handle compile_mode with backwards compatibility
        if compile_mode is not None:
            self._compile_mode = compile_mode
        elif compile:
            # Legacy: compile=True -> FULL mode
            self._compile_mode = CompileMode.FULL
        else:
            self._compile_mode = CompileMode.EAGER

        # Track compile state for AUTO mode fallback
        self._compile_attempts = []  # History of failed modes
        self._compiled_forward = None  # Cached compiled forward function
        self._compiled_loss_and_grad = None  # Cached compiled loss+grad

        # Track training state
        self.epoch = 0
        self.history = {"loss": [], "compile_fallbacks": []}

        if derivative_method == "autodiff":
            # Functional API approach
            from .functional import FunctionalOptimizer, get_params_list, set_params_list

            self._get_params = get_params_list
            self._set_params = set_params_list
            self.params = get_params_list(model)

            self.optimizer = FunctionalOptimizer(
                self.params,
                lr=lr,
                optimizer=optimizer,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
            )

            # Build derivative functions (functional API - takes params)
            self._u_fn, self._du_fn, self._d2u_fn = make_derivative_fns(model.k, model.base_fun)
            self._uses_params = True

        elif derivative_method == "finite_diff":
            # Model-based approach with finite differences
            self._u_fn, self._du_fn, self._d2u_fn = make_finite_diff_derivative_fns(
                model, h=finite_diff_h
            )
            self._uses_params = False

            # Use standard MLX optimizer on model (or L-BFGS)
            self._optimizer_type = optimizer.lower()
            if optimizer == "adam":
                self._mlx_optimizer = optim.Adam(learning_rate=lr, betas=[beta1, beta2])
            elif optimizer == "adamw":
                self._mlx_optimizer = optim.AdamW(
                    learning_rate=lr, betas=[beta1, beta2], weight_decay=weight_decay
                )
            elif optimizer == "sgd":
                self._mlx_optimizer = optim.SGD(learning_rate=lr)
            elif optimizer == "lbfgs":
                # L-BFGS handled specially in step()
                self._mlx_optimizer = None
                self._lbfgs_max_iter = 20  # Default iterations per step
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}. Use 'adam', 'adamw', 'sgd', or 'lbfgs'")

            self._loss_and_grad_fn = None  # Will be set on first step

        else:
            raise ValueError(
                f"Unknown derivative_method: {derivative_method}. Use 'autodiff' or 'finite_diff'"
            )

    @property
    def lr(self) -> float:
        """Current learning rate."""
        return self._lr

    @lr.setter
    def lr(self, value: float):
        """Set learning rate dynamically.

        Example:
            trainer.lr = 0.0003  # Reduce LR for fine-tuning phase
        """
        self._lr = value
        if self._uses_params:
            self.optimizer.lr = value
        else:
            self._mlx_optimizer.learning_rate = value

    def u(self, *args) -> mx.array:
        """Compute u(x). Returns shape (N,).

        For autodiff: u(params, x)
        For finite_diff: u(x)
        """
        if self._uses_params:
            params, x = args
            return self._u_fn(params, x)
        else:
            (x,) = args
            return self._u_fn(x)

    def du(self, *args) -> mx.array:
        """Compute du/dx. Returns shape (N,) for 1D or (N, dim) for multi-D.

        For autodiff: du(params, x)
        For finite_diff: du(x)
        """
        if self._uses_params:
            params, x = args
            return self._du_fn(params, x)
        else:
            (x,) = args
            return self._du_fn(x)

    def d2u(self, *args) -> mx.array:
        """Compute d²u/dx². Returns shape (N,) for 1D or (N, dim) for multi-D.

        For autodiff: d2u(params, x)
        For finite_diff: d2u(x)
        """
        if self._uses_params:
            params, x = args
            return self._d2u_fn(params, x)
        else:
            (x,) = args
            return self._d2u_fn(x)

    def laplacian(self, *args) -> mx.array:
        """Compute Laplacian ∇²u = Σᵢ d²u/dxᵢ². Returns shape (N,).

        For autodiff: laplacian(params, x)
        For finite_diff: laplacian(x)
        """
        if self._uses_params:
            params, x = args
            d2u = self._d2u_fn(params, x)
        else:
            (x,) = args
            d2u = self._d2u_fn(x)

        if len(d2u.shape) == 1 or d2u.shape[-1] == 1:
            return mx.squeeze(d2u) if len(d2u.shape) > 1 else d2u
        return mx.sum(d2u, axis=-1)

    def step(self, loss_fn: Callable, *loss_args) -> float:
        """Perform one training step.

        Args:
            loss_fn: Loss function.
                - For autodiff: (params, *args) -> scalar
                - For finite_diff: (*args) -> scalar OR (model, *args) -> scalar
                  If first param is not 'model', the function is wrapped automatically.
            *loss_args: Additional arguments passed to loss_fn

        Returns:
            Loss value as float

        Example (finite_diff mode - no model arg needed):
            def my_loss(x, alpha):
                u = trainer.u(x)
                return mx.mean((u - target)**2) * alpha

            trainer.step(my_loss, x_train, alpha=1.0)
        """
        import inspect

        import mlx.nn as nn

        if self._uses_params:
            # Functional API (autodiff)
            loss_val, grads = self._step_functional(loss_fn, *loss_args)

            # Update parameters
            self.params = self.optimizer.update(self.params, grads)

            # Evaluate
            mx.eval(self.params, self.optimizer.state, loss_val)
        else:
            # Model-based (finite_diff)
            # Auto-detect if loss_fn expects 'model' as first arg
            sig = inspect.signature(loss_fn)
            params = list(sig.parameters.keys())
            first_param = params[0] if params else None

            if first_param != "model":
                # Wrap loss_fn to add model arg (required by nn.value_and_grad)
                original_fn = loss_fn

                def wrapped_loss_fn(model, *args):
                    return original_fn(*args)
            else:
                wrapped_loss_fn = loss_fn
                original_fn = loss_fn

            # Handle L-BFGS specially
            if self._optimizer_type == "lbfgs":
                # L-BFGS: run fit_lbfgs with current max_iter setting
                # For L-BFGS, loss_args[0] should be x_train
                if not loss_args:
                    raise ValueError("L-BFGS requires training data as first argument to step()")

                x_train = loss_args[0]
                # Create a simple loss that only takes x
                def simple_loss(x):
                    return original_fn(x, *loss_args[1:]) if len(loss_args) > 1 else original_fn(x)

                result = self.fit_lbfgs(
                    simple_loss,
                    x_train,
                    max_iter=self._lbfgs_max_iter,
                    verbose=False,
                )
                loss_val = mx.array(result["loss"])
            else:
                # Standard first-order optimizers (Adam, AdamW, SGD)
                loss_val, grads = self._step_model_based(wrapped_loss_fn, *loss_args)

                # Update model
                self._mlx_optimizer.update(self.model, grads)

                # Evaluate
                mx.eval(self.model.parameters(), self._mlx_optimizer.state, loss_val)

        self.epoch += 1
        loss_float = float(loss_val)
        self.history["loss"].append(loss_float)

        return loss_float

    def _step_functional(
        self, loss_fn: Callable, *loss_args
    ) -> tuple[mx.array, ParamsList]:
        """Execute training step with functional API and safe compile modes.

        Implements the compile mode logic:
        - EAGER: No compilation
        - FORWARD_ONLY: Compile forward pass, grad stays outside (recommended)
        - FULL: Compile entire loss_and_grad (fastest, may fail)
        - AUTO: Try FULL -> FORWARD_ONLY -> EAGER with fallback

        Returns:
            Tuple of (loss_value, gradients)
        """
        mode = self._compile_mode

        # AUTO mode: try modes in order until one works
        if mode == CompileMode.AUTO:
            modes_to_try = [CompileMode.FULL, CompileMode.FORWARD_ONLY, CompileMode.EAGER]
            # Skip modes that already failed
            modes_to_try = [m for m in modes_to_try if m not in self._compile_attempts]

            for try_mode in modes_to_try:
                try:
                    return self._execute_step_with_mode(loss_fn, try_mode, *loss_args)
                except (ValueError, RuntimeError) as e:
                    # VJP failure or compile error - record and try next
                    self._compile_attempts.append(try_mode)
                    self.history["compile_fallbacks"].append({
                        "step": self.epoch,
                        "failed_mode": try_mode.value,
                        "error": str(e)[:100],
                    })

            # All modes failed - this shouldn't happen (EAGER should always work)
            raise RuntimeError("All compile modes failed, including EAGER")

        # Specific mode requested
        return self._execute_step_with_mode(loss_fn, mode, *loss_args)

    def _execute_step_with_mode(
        self, loss_fn: Callable, mode: CompileMode, *loss_args
    ) -> tuple[mx.array, ParamsList]:
        """Execute step with specific compile mode."""

        if mode == CompileMode.EAGER:
            # No compilation - safest, always works
            loss_and_grad = mx.value_and_grad(loss_fn)
            return loss_and_grad(self.params, *loss_args)

        elif mode == CompileMode.FORWARD_ONLY:
            # Compile forward pass only - recommended for most cases
            # The forward runs on Metal, gradients use standard rules
            if self._compiled_forward is None:
                @mx.compile
                def compiled_loss(params, *args):
                    return loss_fn(params, *args)
                self._compiled_forward = compiled_loss

            # Wrap compiled forward with grad (grad is NOT compiled)
            loss_and_grad = mx.value_and_grad(self._compiled_forward)
            return loss_and_grad(self.params, *loss_args)

        elif mode == CompileMode.FULL:
            # Compile entire loss_and_grad - fastest but may hit VJP issues
            if self._compiled_loss_and_grad is None:
                loss_and_grad = mx.value_and_grad(loss_fn)
                self._compiled_loss_and_grad = mx.compile(loss_and_grad)

            return self._compiled_loss_and_grad(self.params, *loss_args)

        else:
            raise ValueError(f"Unknown compile mode: {mode}")

    def _step_model_based(
        self, wrapped_loss_fn: Callable, *loss_args
    ) -> tuple[mx.array, Any]:
        """Execute training step for model-based (finite_diff) mode with safe compile.

        Returns:
            Tuple of (loss_value, gradients)
        """
        import mlx.nn as nn

        mode = self._compile_mode

        if mode == CompileMode.EAGER:
            loss_and_grad = nn.value_and_grad(self.model, wrapped_loss_fn)
            return loss_and_grad(self.model, *loss_args)

        elif mode == CompileMode.FORWARD_ONLY:
            # Compile forward, grad outside
            if self._compiled_forward is None:
                @mx.compile
                def compiled_loss(model, *args):
                    return wrapped_loss_fn(model, *args)
                self._compiled_forward = compiled_loss

            loss_and_grad = nn.value_and_grad(self.model, self._compiled_forward)
            return loss_and_grad(self.model, *loss_args)

        elif mode == CompileMode.FULL:
            if self._compiled_loss_and_grad is None:
                loss_and_grad = nn.value_and_grad(self.model, wrapped_loss_fn)
                self._compiled_loss_and_grad = mx.compile(loss_and_grad)

            try:
                return self._compiled_loss_and_grad(self.model, *loss_args)
            except (ValueError, RuntimeError) as e:
                # Fallback to non-compiled
                self._compiled_loss_and_grad = False
                loss_and_grad = nn.value_and_grad(self.model, wrapped_loss_fn)
                return loss_and_grad(self.model, *loss_args)

        elif mode == CompileMode.AUTO:
            # Try FULL first, fallback to FORWARD_ONLY, then EAGER
            for try_mode in [CompileMode.FULL, CompileMode.FORWARD_ONLY, CompileMode.EAGER]:
                if try_mode in self._compile_attempts:
                    continue
                try:
                    return self._step_model_based_with_mode(wrapped_loss_fn, try_mode, *loss_args)
                except (ValueError, RuntimeError):
                    self._compile_attempts.append(try_mode)

            raise RuntimeError("All compile modes failed")

        else:
            raise ValueError(f"Unknown compile mode: {mode}")

    def _step_model_based_with_mode(
        self, wrapped_loss_fn: Callable, mode: CompileMode, *loss_args
    ) -> tuple[mx.array, Any]:
        """Helper for model-based step with specific mode."""
        import mlx.nn as nn

        if mode == CompileMode.EAGER:
            loss_and_grad = nn.value_and_grad(self.model, wrapped_loss_fn)
            return loss_and_grad(self.model, *loss_args)

        elif mode == CompileMode.FORWARD_ONLY:
            @mx.compile
            def compiled_loss(model, *args):
                return wrapped_loss_fn(model, *args)
            loss_and_grad = nn.value_and_grad(self.model, compiled_loss)
            return loss_and_grad(self.model, *loss_args)

        elif mode == CompileMode.FULL:
            loss_and_grad = nn.value_and_grad(self.model, wrapped_loss_fn)
            compiled = mx.compile(loss_and_grad)
            return compiled(self.model, *loss_args)

        raise ValueError(f"Unknown mode: {mode}")

    def train(
        self,
        loss_fn: Callable,
        data_fn: Callable[[], mx.array],
        epochs: int,
        log_freq: int = 100,
        callbacks: list[Callable] | None = None,
    ) -> dict[str, list[float]]:
        """Full training loop.

        Args:
            loss_fn: Loss function
                - For autodiff: (params, x) -> scalar
                - For finite_diff: (model, x) -> scalar
            data_fn: Function that returns training data (called each epoch)
            epochs: Number of epochs
            log_freq: Frequency of logging
            callbacks: Optional list of callback functions

        Returns:
            Training history dictionary
        """
        for epoch in range(epochs):
            x_batch = data_fn()
            loss = self.step(loss_fn, x_batch)

            if epoch % log_freq == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:5d}/{epochs}: loss = {loss:.6f}")

            if callbacks:
                for cb in callbacks:
                    cb(self, epoch, loss)

        # Sync model (only needed for autodiff mode)
        if self._uses_params:
            self.sync_model()

        return self.history

    def sync_model(self):
        """Update the model with current parameters (autodiff mode only)."""
        if self._uses_params:
            self._set_params(self.model, self.params)

    def get_params(self) -> ParamsList:
        """Get current parameters (autodiff mode only)."""
        if self._uses_params:
            return self.params
        raise RuntimeError("get_params() not available in finite_diff mode. Access model directly.")

    def set_params(self, params: ParamsList):
        """Set parameters manually (autodiff mode only)."""
        if self._uses_params:
            self.params = params
        else:
            raise RuntimeError("set_params() not available in finite_diff mode. Access model directly.")

    def fit_lbfgs(
        self,
        loss_fn: Callable,
        x_train: mx.array,
        max_iter: int = 100,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 10,
        verbose: bool = True,
    ) -> dict:
        """Optimize using L-BFGS (second-order method).

        L-BFGS often converges faster than Adam for KANs, especially for
        physics problems where high precision is needed.

        Note: L-BFGS uses fixed training data (no mini-batches).

        Args:
            loss_fn: Loss function (x) -> scalar (no model/params arg needed)
            x_train: Training data (fixed for all iterations)
            max_iter: Maximum L-BFGS iterations
            tolerance_grad: Gradient norm tolerance for convergence
            tolerance_change: Function value change tolerance
            history_size: L-BFGS memory (number of past gradients)
            verbose: Print progress

        Returns:
            Dictionary with 'loss', 'success', 'message', 'n_iter'

        Example:
            def pde_loss(x):
                u = trainer.u(x)
                u_xx = trainer.d2u(x)
                return mx.mean((u_xx + u)**2)

            result = trainer.fit_lbfgs(pde_loss, x_train, max_iter=100)
            print(f"Final loss: {result['loss']:.6f}")
        """
        from .functional import (
            LayerParams,
            flatten_params,
            functional_forward,
            get_params_list,
            set_params_list,
            unflatten_params,
        )

        import numpy as np
        from scipy.optimize import minimize

        # Get initial parameters from model
        params_list = get_params_list(self.model)
        flat_init, metadata = flatten_params(params_list)

        iteration = [0]
        losses = []

        # Build a functional loss that uses params
        def params_loss(params):
            # Update model with these params so trainer.u/du/d2u work
            set_params_list(self.model, params)
            return loss_fn(x_train)

        loss_and_grad_fn = mx.value_and_grad(params_loss)

        def objective_and_grad(flat_p: np.ndarray):
            """Compute loss and gradient for scipy."""
            # Reconstruct params from flat array
            params = unflatten_params(flat_p, metadata)

            # Compute loss and gradients w.r.t. params
            loss, grads = loss_and_grad_fn(params)
            mx.eval(loss, grads)

            # Flatten gradients
            flat_grads, _ = flatten_params(grads)

            iteration[0] += 1
            loss_val = float(loss)
            losses.append(loss_val)

            if verbose and iteration[0] % 10 == 0:
                print(f"  L-BFGS iter {iteration[0]:4d}: loss = {loss_val:.6f}")

            return loss_val, flat_grads

        if verbose:
            print(f"Starting L-BFGS optimization (max_iter={max_iter})...")

        result = minimize(
            objective_and_grad,
            flat_init,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxiter": max_iter,
                "gtol": tolerance_grad,
                "ftol": tolerance_change,
                "maxcor": history_size,
                "disp": False,
            },
        )

        # Apply final parameters to model
        final_params = unflatten_params(result.x, metadata)
        set_params_list(self.model, final_params)

        if verbose:
            print(f"L-BFGS completed: {result.message}")
            print(f"  Final loss: {result.fun:.6f}, iterations: {result.nit}")

        self.history["loss"].extend(losses)

        return {
            "loss": result.fun,
            "success": result.success,
            "message": result.message,
            "n_iter": result.nit,
            "losses": losses,
        }


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
