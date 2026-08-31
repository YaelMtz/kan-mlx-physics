"""Pure functional interface for KAN-MLX-Physics.

This module provides stateless, pure functions for KAN operations that work
seamlessly with mx.grad, mx.vmap, and mx.compile. Use these for physics-informed
neural networks (PINNs) and other applications requiring automatic differentiation.

Example:
    from kan_mlx_physics import MultKAN
    from kan_mlx_physics.functional import functional_forward, get_params_list

    model = MultKAN(width=[1, 2, 1], grid=5, k=3)
    params = get_params_list(model)

    # Pure forward - works with mx.grad
    y = functional_forward(params, x, model.k, model.base_fun)

LBFGS Optimization:
    from kan_mlx_physics.functional import lbfgs_optimize

    def loss_fn(params):
        y = functional_forward(params, x, k, base_fun)
        return mx.mean((y - target)**2)

    final_params, result = lbfgs_optimize(loss_fn, params, max_iter=100)
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import mlx.core as mx
import numpy as np

from .spline import B_batch

if TYPE_CHECKING:
    from .multkan import MultKAN


class LayerParams(NamedTuple):
    """Parameters for a single KAN layer.

    This NamedTuple provides named access to layer parameters,
    improving code readability and IDE autocomplete support.

    Attributes:
        coef: Spline coefficients of shape (in_dim, out_dim, num_coef)
        scale_sp: Spline scaling factors of shape (in_dim, out_dim)
        scale_base: Base function scaling factors of shape (in_dim, out_dim)
        grid: Knot positions of shape (in_dim, num_grid + 2*k + 1)
    """

    coef: mx.array
    scale_sp: mx.array
    scale_base: mx.array
    grid: mx.array


# Type alias for list of layer parameters
ParamsList = list[LayerParams]

# Legacy alias for backward compatibility
ParamsTuple = tuple[mx.array, mx.array, mx.array, mx.array]


def kan_layer_forward(
    x: mx.array,
    coef: mx.array,
    scale_sp: mx.array,
    scale_base: mx.array,
    grid: mx.array,
    k: int,
    base_fun: Callable[[mx.array], mx.array],
) -> mx.array:
    """Pure functional KAN layer forward pass.

    This function has no side effects and can be used with mx.grad.

    Args:
        x: Input tensor of shape (batch, in_dim)
        coef: Spline coefficients of shape (in_dim, out_dim, num_coef)
        scale_sp: Spline scale of shape (in_dim, out_dim)
        scale_base: Base function scale of shape (in_dim, out_dim)
        grid: Knot positions of shape (in_dim, num_grid + 2*k + 1)
        k: Spline order
        base_fun: Base activation function (e.g., identity, silu)

    Returns:
        Output tensor of shape (batch, out_dim)
    """
    # Compute B-spline basis: (batch, in_dim, num_coef)
    bases = B_batch(x, grid, k)

    # Spline output: sum over coefficients
    # bases: (batch, in_dim, num_coef) -> (batch, in_dim, 1, num_coef)
    # coef: (in_dim, out_dim, num_coef) -> (1, in_dim, out_dim, num_coef)
    spline_out = mx.sum(bases[:, :, None, :] * coef[None, :, :, :], axis=-1)
    # spline_out shape: (batch, in_dim, out_dim)

    # Base function contribution
    # base_fun(x): (batch, in_dim) -> (batch, in_dim, 1)
    # scale_base: (in_dim, out_dim) -> (1, in_dim, out_dim)
    base_out = base_fun(x)[:, :, None] * scale_base[None, :, :]
    # base_out shape: (batch, in_dim, out_dim)

    # Combined output with scales
    # scale_sp: (in_dim, out_dim) -> (1, in_dim, out_dim)
    y = scale_sp[None, :, :] * spline_out + base_out
    # y shape: (batch, in_dim, out_dim)

    # Sum over input dimension to get final output
    return mx.sum(y, axis=1)  # (batch, out_dim)


def functional_forward(
    params_list: ParamsList,
    x: mx.array,
    k: int,
    base_fun: Callable[[mx.array], mx.array],
) -> mx.array:
    """Pure functional forward pass through entire KAN network.

    This function has no side effects and works with mx.grad, mx.vmap, mx.compile.

    Args:
        params_list: List of parameter tuples [(coef, scale_sp, scale_base, grid), ...]
        x: Input tensor of shape (batch, in_dim)
        k: Spline order
        base_fun: Base activation function

    Returns:
        Output tensor of shape (batch, out_dim)

    Example:
        params = get_params_list(model)
        y = functional_forward(params, x, model.k, model.base_fun)

        # With mx.grad
        def loss_fn(params, x):
            y = functional_forward(params, x, k, base_fun)
            return mx.mean(y**2)

        grads = mx.grad(loss_fn)(params, x)
    """
    for coef, scale_sp, scale_base, grid in params_list:
        x = kan_layer_forward(x, coef, scale_sp, scale_base, grid, k, base_fun)
    return x


def get_params_list(model: "MultKAN") -> ParamsList:
    """Extract parameters from model as a list of LayerParams.

    This format is compatible with functional_forward and mx.grad.

    Args:
        model: MultKAN model instance

    Returns:
        List of LayerParams(coef, scale_sp, scale_base, grid), one per layer
    """
    params = []
    for layer in model.layers:
        # Handle None grid (for non-spline bases like Laguerre)
        # by using a placeholder empty array
        grid = layer.grid if layer.grid is not None else mx.array([0.0])
        params.append(LayerParams(layer.coef, layer.scale_sp, layer.scale_base, grid))
    return params


def set_params_list(model: "MultKAN", params_list: ParamsList) -> None:
    """Update model parameters from a list of LayerParams.

    Args:
        model: MultKAN model instance
        params_list: List of LayerParams(coef, scale_sp, scale_base, grid)
    """
    for i, params in enumerate(params_list):
        model.layers[i].coef = params.coef
        model.layers[i].scale_sp = params.scale_sp
        model.layers[i].scale_base = params.scale_base
        # Note: grid is typically not updated during training


def init_adam_state(params_list: ParamsList) -> tuple[ParamsList, ParamsList]:
    """Initialize Adam optimizer state (first and second moments).

    Args:
        params_list: List of parameter tuples

    Returns:
        Tuple of (m_params, v_params) where each is a list of moment tuples
    """
    m_params = []
    v_params = []

    for coef, scale_sp, scale_base, grid in params_list:
        m_params.append(
            (
                mx.zeros_like(coef),
                mx.zeros_like(scale_sp),
                mx.zeros_like(scale_base),
            )
        )
        v_params.append(
            (
                mx.zeros_like(coef),
                mx.zeros_like(scale_sp),
                mx.zeros_like(scale_base),
            )
        )

    return m_params, v_params


def adam_update(
    params_list: ParamsList,
    grads_list: ParamsList,
    m_params: list[tuple[mx.array, mx.array, mx.array]],
    v_params: list[tuple[mx.array, mx.array, mx.array]],
    t: mx.array,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[ParamsList, list, list]:
    """Perform Adam optimizer update on parameters.

    This is a pure function suitable for use inside @mx.compile.

    Args:
        params_list: Current parameters
        grads_list: Gradients (same structure as params_list)
        m_params: First moment estimates
        v_params: Second moment estimates
        t: Current timestep (1-indexed)
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Small constant for numerical stability

    Returns:
        Tuple of (new_params_list, new_m_params, new_v_params)
    """
    # Bias correction factors
    bc1 = 1.0 - beta1**t
    bc2 = 1.0 - beta2**t

    new_params_list = []
    new_m_params = []
    new_v_params = []

    for i, ((coef, scale_sp, scale_base, grid), (g_coef, g_scale_sp, g_scale_base, _)) in enumerate(
        zip(params_list, grads_list)
    ):
        m_coef, m_sp, m_base = m_params[i]
        v_coef, v_sp, v_base = v_params[i]

        # Update coef
        m_coef = beta1 * m_coef + (1 - beta1) * g_coef
        v_coef = beta2 * v_coef + (1 - beta2) * (g_coef**2)
        m_hat = m_coef / bc1
        v_hat = v_coef / bc2
        new_coef = coef - lr * m_hat / (mx.sqrt(v_hat) + eps)

        # Update scale_sp
        m_sp = beta1 * m_sp + (1 - beta1) * g_scale_sp
        v_sp = beta2 * v_sp + (1 - beta2) * (g_scale_sp**2)
        m_hat = m_sp / bc1
        v_hat = v_sp / bc2
        new_scale_sp = scale_sp - lr * m_hat / (mx.sqrt(v_hat) + eps)

        # Update scale_base
        m_base = beta1 * m_base + (1 - beta1) * g_scale_base
        v_base = beta2 * v_base + (1 - beta2) * (g_scale_base**2)
        m_hat = m_base / bc1
        v_hat = v_base / bc2
        new_scale_base = scale_base - lr * m_hat / (mx.sqrt(v_hat) + eps)

        new_params_list.append((new_coef, new_scale_sp, new_scale_base, grid))
        new_m_params.append((m_coef, m_sp, m_base))
        new_v_params.append((v_coef, v_sp, v_base))

    return new_params_list, new_m_params, new_v_params


# ============================================================
# LBFGS Optimizer
# ============================================================


def flatten_params(params_list: ParamsList) -> tuple[np.ndarray, dict[str, Any]]:
    """Flatten params_list to 1D numpy array.

    Args:
        params_list: List of (coef, scale_sp, scale_base, grid) tuples

    Returns:
        Tuple of (flat_array, metadata) where metadata contains shapes for reconstruction
    """
    flat_parts = []
    metadata = {
        "shapes": [],
        "grids": [],
        "n_layers": len(params_list),
    }

    for coef, scale_sp, scale_base, grid in params_list:
        # Store shapes
        metadata["shapes"].append(
            {
                "coef": coef.shape,
                "scale_sp": scale_sp.shape,
                "scale_base": scale_base.shape,
            }
        )
        # Store grids (not optimized)
        metadata["grids"].append(np.array(grid))

        # Flatten trainable parameters
        flat_parts.append(np.array(coef).flatten())
        flat_parts.append(np.array(scale_sp).flatten())
        flat_parts.append(np.array(scale_base).flatten())

    return np.concatenate(flat_parts), metadata


def unflatten_params(flat: np.ndarray, metadata: dict[str, Any]) -> ParamsList:
    """Reconstruct params_list from flattened array.

    Args:
        flat: 1D numpy array of parameters
        metadata: Dictionary with shapes and grids

    Returns:
        List of (coef, scale_sp, scale_base, grid) tuples
    """
    params_list = []
    offset = 0

    for i in range(metadata["n_layers"]):
        shapes = metadata["shapes"][i]
        grid = mx.array(metadata["grids"][i])

        # Reconstruct coef
        coef_size = np.prod(shapes["coef"])
        coef = mx.array(flat[offset : offset + coef_size].reshape(shapes["coef"]))
        offset += coef_size

        # Reconstruct scale_sp
        sp_size = np.prod(shapes["scale_sp"])
        scale_sp = mx.array(flat[offset : offset + sp_size].reshape(shapes["scale_sp"]))
        offset += sp_size

        # Reconstruct scale_base
        base_size = np.prod(shapes["scale_base"])
        scale_base = mx.array(flat[offset : offset + base_size].reshape(shapes["scale_base"]))
        offset += base_size

        params_list.append(LayerParams(coef, scale_sp, scale_base, grid))

    return params_list


def lbfgs_optimize(
    loss_fn: Callable[[ParamsList], mx.array],
    params_list: ParamsList,
    max_iter: int = 100,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    history_size: int = 10,
    callback: Callable[[np.ndarray, int], None] | None = None,
    verbose: bool = False,
) -> tuple[ParamsList, Any]:
    """L-BFGS optimization for KAN parameters.

    Uses scipy.optimize.minimize with L-BFGS-B method for quasi-Newton
    optimization. Often converges faster than Adam for physics problems.

    Args:
        loss_fn: Loss function taking params_list and returning scalar loss
        params_list: Initial parameters
        max_iter: Maximum number of iterations
        tolerance_grad: Gradient tolerance for convergence
        tolerance_change: Function value change tolerance
        history_size: Number of past gradients to store (L-BFGS memory)
        callback: Optional callback(flat_params, iteration) called each step
        verbose: Print progress information

    Returns:
        Tuple of (optimized_params_list, scipy_result)

    Example:
        def loss_fn(params):
            y = functional_forward(params, x, k, base_fun)
            return mx.mean((y - target)**2)

        params, result = lbfgs_optimize(loss_fn, initial_params, max_iter=100)
        print(f"Final loss: {result.fun}")
    """
    from scipy.optimize import minimize

    # Flatten initial parameters
    flat_init, metadata = flatten_params(params_list)

    # Track iteration for callback
    iteration = [0]

    def objective(flat_p: np.ndarray) -> float:
        """Compute loss value."""
        params = unflatten_params(flat_p, metadata)
        loss = loss_fn(params)
        mx.eval(loss)
        return float(loss)

    def gradient(flat_p: np.ndarray) -> np.ndarray:
        """Compute gradient of loss."""
        params = unflatten_params(flat_p, metadata)

        # Compute loss and gradients
        loss_and_grad = mx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad(params)
        mx.eval(loss, grads)

        # Flatten gradients
        flat_grads, _ = flatten_params(grads)

        return flat_grads

    def scipy_callback(xk: np.ndarray):
        """Callback for scipy optimizer."""
        iteration[0] += 1
        if callback is not None:
            callback(xk, iteration[0])
        if verbose and iteration[0] % 10 == 0:
            loss = objective(xk)
            print(f"  LBFGS iter {iteration[0]}: loss = {loss:.6f}")

    if verbose:
        print(f"Starting L-BFGS optimization (max_iter={max_iter})")

    # Run optimization
    result = minimize(
        objective,
        flat_init,
        method="L-BFGS-B",
        jac=gradient,
        callback=scipy_callback,
        options={
            "maxiter": max_iter,
            "gtol": tolerance_grad,
            "ftol": tolerance_change,
            "maxcor": history_size,
            "disp": False,
        },
    )

    if verbose:
        print(f"L-BFGS finished: {result.message}")
        print(f"  Final loss: {result.fun:.6f}")
        print(f"  Iterations: {result.nit}")
        print(f"  Function evaluations: {result.nfev}")

    # Reconstruct final parameters
    final_params = unflatten_params(result.x, metadata)

    return final_params, result


# ============================================================
# Functional Optimizer (works with params lists)
# ============================================================


class FunctionalOptimizer:
    """Optimizer that works with the functional API's params list format.

    This bridges the gap between mlx.optimizers and the functional API,
    providing a familiar interface while supporting params lists.

    Example:
        from kan_mlx_physics.functional import get_params_list, FunctionalOptimizer

        model = MultKAN(width=[1, 5, 1])
        params = get_params_list(model)

        optimizer = FunctionalOptimizer(params, lr=0.01, optimizer='adam')

        for epoch in range(100):
            loss, grads = loss_and_grad(params, x)
            params = optimizer.update(params, grads)

        set_params_list(model, params)
    """

    def __init__(
        self,
        params: ParamsList,
        lr: float = 0.001,
        optimizer: str = "adam",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
    ):
        """Initialize the functional optimizer.

        Args:
            params: Initial params list from get_params_list()
            lr: Learning rate
            optimizer: Optimizer type ('adam', 'sgd', 'adamw')
            beta1: Adam beta1 (momentum decay)
            beta2: Adam beta2 (RMSprop decay)
            eps: Small constant for numerical stability
            weight_decay: L2 regularization weight (for adamw)
            momentum: Momentum for SGD
        """
        self.lr = lr
        self.optimizer_type = optimizer.lower()
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.t = 0

        # Initialize optimizer state
        self._init_state(params)

    def _init_state(self, params: ParamsList):
        """Initialize optimizer state based on optimizer type."""
        if self.optimizer_type in ("adam", "adamw"):
            # First moment (m) and second moment (v) for Adam
            self.m = []
            self.v = []
            for coef, scale_sp, scale_base, grid in params:
                self.m.append(
                    (
                        mx.zeros_like(coef),
                        mx.zeros_like(scale_sp),
                        mx.zeros_like(scale_base),
                    )
                )
                self.v.append(
                    (
                        mx.zeros_like(coef),
                        mx.zeros_like(scale_sp),
                        mx.zeros_like(scale_base),
                    )
                )
        elif self.optimizer_type == "sgd":
            # Velocity for momentum SGD
            self.velocity = []
            for coef, scale_sp, scale_base, grid in params:
                self.velocity.append(
                    (
                        mx.zeros_like(coef),
                        mx.zeros_like(scale_sp),
                        mx.zeros_like(scale_base),
                    )
                )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

    @property
    def state(self) -> list:
        """Return optimizer state for mx.eval().

        Returns a flat list of all state tensors (m, v for Adam, velocity for SGD).
        """
        state_tensors = []
        if self.optimizer_type in ("adam", "adamw"):
            for m_tuple, v_tuple in zip(self.m, self.v):
                state_tensors.extend(m_tuple)
                state_tensors.extend(v_tuple)
        elif self.optimizer_type == "sgd":
            for v_tuple in self.velocity:
                state_tensors.extend(v_tuple)
        return state_tensors

    def update(self, params: ParamsList, grads: ParamsList) -> ParamsList:
        """Update parameters using gradients.

        Args:
            params: Current parameters
            grads: Gradients (same structure as params)

        Returns:
            Updated parameters
        """
        self.t += 1

        if self.optimizer_type == "adam":
            return self._adam_update(params, grads)
        elif self.optimizer_type == "adamw":
            return self._adamw_update(params, grads)
        elif self.optimizer_type == "sgd":
            return self._sgd_update(params, grads)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

    def _adam_update(self, params: ParamsList, grads: ParamsList) -> ParamsList:
        """Adam optimizer update."""
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t

        new_params = []
        new_m = []
        new_v = []

        for i, ((coef, scale_sp, scale_base, grid), (g_coef, g_sp, g_base, _)) in enumerate(
            zip(params, grads)
        ):
            m_coef, m_sp, m_base = self.m[i]
            v_coef, v_sp, v_base = self.v[i]

            # Update coef
            m_coef = self.beta1 * m_coef + (1 - self.beta1) * g_coef
            v_coef = self.beta2 * v_coef + (1 - self.beta2) * (g_coef**2)
            new_coef = coef - self.lr * (m_coef / bc1) / (mx.sqrt(v_coef / bc2) + self.eps)

            # Update scale_sp
            m_sp = self.beta1 * m_sp + (1 - self.beta1) * g_sp
            v_sp = self.beta2 * v_sp + (1 - self.beta2) * (g_sp**2)
            new_sp = scale_sp - self.lr * (m_sp / bc1) / (mx.sqrt(v_sp / bc2) + self.eps)

            # Update scale_base
            m_base = self.beta1 * m_base + (1 - self.beta1) * g_base
            v_base = self.beta2 * v_base + (1 - self.beta2) * (g_base**2)
            new_base = scale_base - self.lr * (m_base / bc1) / (mx.sqrt(v_base / bc2) + self.eps)

            new_params.append(LayerParams(new_coef, new_sp, new_base, grid))
            new_m.append((m_coef, m_sp, m_base))
            new_v.append((v_coef, v_sp, v_base))

        self.m = new_m
        self.v = new_v
        return new_params

    def _adamw_update(self, params: ParamsList, grads: ParamsList) -> ParamsList:
        """AdamW optimizer update (Adam with decoupled weight decay)."""
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t

        new_params = []
        new_m = []
        new_v = []

        for i, ((coef, scale_sp, scale_base, grid), (g_coef, g_sp, g_base, _)) in enumerate(
            zip(params, grads)
        ):
            m_coef, m_sp, m_base = self.m[i]
            v_coef, v_sp, v_base = self.v[i]

            # Weight decay
            coef = coef * (1 - self.lr * self.weight_decay)
            scale_sp = scale_sp * (1 - self.lr * self.weight_decay)
            scale_base = scale_base * (1 - self.lr * self.weight_decay)

            # Update coef
            m_coef = self.beta1 * m_coef + (1 - self.beta1) * g_coef
            v_coef = self.beta2 * v_coef + (1 - self.beta2) * (g_coef**2)
            new_coef = coef - self.lr * (m_coef / bc1) / (mx.sqrt(v_coef / bc2) + self.eps)

            # Update scale_sp
            m_sp = self.beta1 * m_sp + (1 - self.beta1) * g_sp
            v_sp = self.beta2 * v_sp + (1 - self.beta2) * (g_sp**2)
            new_sp = scale_sp - self.lr * (m_sp / bc1) / (mx.sqrt(v_sp / bc2) + self.eps)

            # Update scale_base
            m_base = self.beta1 * m_base + (1 - self.beta1) * g_base
            v_base = self.beta2 * v_base + (1 - self.beta2) * (g_base**2)
            new_base = scale_base - self.lr * (m_base / bc1) / (mx.sqrt(v_base / bc2) + self.eps)

            new_params.append(LayerParams(new_coef, new_sp, new_base, grid))
            new_m.append((m_coef, m_sp, m_base))
            new_v.append((v_coef, v_sp, v_base))

        self.m = new_m
        self.v = new_v
        return new_params

    def _sgd_update(self, params: ParamsList, grads: ParamsList) -> ParamsList:
        """SGD with momentum update."""
        new_params = []
        new_velocity = []

        for i, ((coef, scale_sp, scale_base, grid), (g_coef, g_sp, g_base, _)) in enumerate(
            zip(params, grads)
        ):
            v_coef, v_sp, v_base = self.velocity[i]

            # Update with momentum
            v_coef = self.momentum * v_coef + g_coef
            v_sp = self.momentum * v_sp + g_sp
            v_base = self.momentum * v_base + g_base

            new_coef = coef - self.lr * v_coef
            new_sp = scale_sp - self.lr * v_sp
            new_base = scale_base - self.lr * v_base

            new_params.append(LayerParams(new_coef, new_sp, new_base, grid))
            new_velocity.append((v_coef, v_sp, v_base))

        self.velocity = new_velocity
        return new_params

    @property
    def state(self):
        """Return optimizer state for mx.eval()."""
        if self.optimizer_type in ("adam", "adamw"):
            return self.m + self.v
        elif self.optimizer_type == "sgd":
            return self.velocity
        return []


def lbfgs_fit(
    model,
    dataset: dict[str, mx.array],
    loss_fn: Callable | None = None,
    max_iter: int = 100,
    lamb: float = 0.0,
    lamb_l1: float = 1.0,
    lamb_entropy: float = 2.0,
    update_grid: bool = True,
    grid_update_freq: int = 20,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Fit model using L-BFGS optimizer.

    Convenience function that handles parameter extraction, optimization,
    and model updates.

    Args:
        model: MultKAN model instance
        dataset: Dictionary with 'train_input' and 'train_label' keys
        loss_fn: Custom loss function (optional)
        max_iter: Maximum iterations
        lamb: Overall regularization weight
        lamb_l1: L1 regularization on activations
        lamb_entropy: Entropy regularization
        update_grid: Whether to update grid during training
        grid_update_freq: Frequency of grid updates
        verbose: Print progress

    Returns:
        Dictionary with training history

    Example:
        dataset = {'train_input': x, 'train_label': y}
        history = lbfgs_fit(model, dataset, max_iter=50)
    """
    x_train = dataset["train_input"]
    y_train = dataset["train_label"]

    k = model.k
    base_fun = model.base_fun

    # Get initial parameters
    params_list = get_params_list(model)

    # Default loss function
    if loss_fn is None:

        def loss_fn(params):
            y_pred = functional_forward(params, x_train, k, base_fun)
            mse = mx.mean((y_pred - y_train) ** 2)

            # Regularization
            reg = mx.array(0.0)
            if lamb > 0:
                for coef, scale_sp, scale_base, grid in params:
                    reg = reg + lamb_l1 * mx.mean(mx.abs(scale_sp))
                    # Entropy regularization
                    if lamb_entropy > 0:
                        p = mx.abs(scale_sp) / (mx.sum(mx.abs(scale_sp)) + 1e-8)
                        reg = reg + lamb_entropy * mx.sum(p * mx.log(p + 1e-8))

            return mse + lamb * reg

    history = {"loss": [], "iteration": []}
    last_grid_update = [0]

    def callback(xk, iteration):
        history["iteration"].append(iteration)

        # Grid update
        if update_grid and iteration - last_grid_update[0] >= grid_update_freq:
            params = unflatten_params(xk, metadata)
            set_params_list(model, params)
            model.update_grid_from_samples(x_train)
            # Re-extract params after grid update
            new_params = get_params_list(model)
            new_flat, _ = flatten_params(new_params)
            xk[:] = new_flat
            last_grid_update[0] = iteration
            if verbose:
                print(f"  Grid updated at iteration {iteration}")

    # Flatten for metadata reference in callback
    _, metadata = flatten_params(params_list)

    # Run optimization
    final_params, result = lbfgs_optimize(
        loss_fn,
        params_list,
        max_iter=max_iter,
        callback=callback,
        verbose=verbose,
    )

    # Update model
    set_params_list(model, final_params)

    history["final_loss"] = float(result.fun)
    history["converged"] = result.success
    history["message"] = result.message

    return history
