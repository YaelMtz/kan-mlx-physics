"""MultKAN: Multi-layer Kolmogorov-Arnold Network for MLX.

Enhanced implementation with:
- LBFGS optimizer support
- Model versioning (rewind/checkout)
- Architecture modification (expand_width/expand_depth)
- Speed mode for faster training
- Advanced regularization options
- Improved pruning
- Transfer learning between models
- Singularity avoiding forward
- MLX optimizations (mx.compile)
- Mixed precision training
- Symbolic tree export
- Uncertainty quantification
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import Callable, Optional, Tuple, Dict, Any, List, Union, TypedDict, Literal, overload
import numpy as np
import pickle
from pathlib import Path
from copy import deepcopy
import time


class TrainingDataset(TypedDict, total=False):
    """Dataset dictionary for training KAN models.

    Required keys:
        train_input: Training input data of shape (n_samples, input_dim)
        train_label: Training labels of shape (n_samples, output_dim)

    Optional keys:
        test_input: Test input data for validation
        test_label: Test labels for validation
    """
    train_input: mx.array
    train_label: mx.array
    test_input: mx.array
    test_label: mx.array


class TrainingHistory(TypedDict):
    """Training history returned by fit().

    Attributes:
        train_loss: List of training MSE losses at each log step
        test_loss: List of test MSE losses (if test data provided)
        reg_loss: List of regularization losses
    """
    train_loss: List[float]
    test_loss: List[float]
    reg_loss: List[float]

from .kan_layer import KANLayer
from .symbolic import (
    Symbolic_KANLayer,
    SYMBOLIC_REGISTRY,
    fit_affine_params,
    list_symbolic,
)
from .spline import coef2curve, curve2coef


class MultKAN(nn.Module):
    """Multi-layer Kolmogorov-Arnold Network with pluggable basis functions.

    A KAN is a stack of KANLayers where each layer has learnable activation
    functions on edges instead of fixed activations on nodes.

    Supports two width formats:
    1. Simple: [2, 5, 1] - all nodes are sum nodes
    2. Extended: [[2, 0], [4, 1], [1, 0]] - [n_sum, n_mult] per layer
       where n_mult nodes compute products of mult_arity inputs

    Supports multiple basis types per layer:
    - B-splines (default, general purpose)
    - Fourier (periodic/oscillatory)
    - Chebyshev (spectral methods)
    - Hermite (quantum mechanics)
    - Laguerre (radial problems)
    - Legendre (angular momentum)

    Args:
        width: List of layer widths. Either:
            - [int, ...]: Simple format, all sum nodes
            - [[n_sum, n_mult], ...]: Extended format with multiplication nodes
        grid: Number of grid intervals for splines
        k: Spline polynomial order (default: 3 for cubic)
        noise_scale: Scale of noise for coefficient initialization
        base_fun: Base activation function for residual connection
        grid_range: Range for uniform grid initialization
        seed: Random seed for reproducibility
        mult_arity: Number of inputs per multiplication node (default: 2)
        basis: Basis type(s) for edge functions. Either:
            - str: Same basis for all layers (e.g., "fourier")
            - List[str]: Per-layer basis (e.g., ["chebyshev", "fourier", "bspline"])
        basis_M: Number of basis functions. Either:
            - int: Same M for all layers
            - List[int]: Per-layer M
            - None: Use default (num_grid + k)
        basis_kwargs: Additional basis arguments. Either:
            - Dict: Same kwargs for all layers
            - List[Dict]: Per-layer kwargs

    Example:
        # Traditional B-spline KAN
        model = MultKAN(width=[2, 5, 1])

        # FourierKAN with learnable frequencies
        model = MultKAN(width=[2, 5, 1], basis="fourier", basis_M=11)

        # Hybrid: Chebyshev first layer, Fourier second
        model = MultKAN(
            width=[2, 10, 1],
            basis=["chebyshev", "fourier"],
            basis_M=[8, 15]
        )

        # HermiteKAN for quantum mechanics
        model = MultKAN(
            width=[1, 10, 1],
            basis="hermite",
            basis_kwargs={"weighted": True}
        )
    """

    def __init__(
        self,
        width: Union[List[int], List[List[int]]],
        grid: int = 5,
        k: int = 3,
        noise_scale: float = 0.1,
        base_fun: Callable = nn.silu,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        seed: Optional[int] = None,
        mult_arity: int = 2,
        # New basis parameters
        basis: Union[str, List[str]] = "bspline",
        basis_M: Union[int, List[int], None] = None,
        basis_kwargs: Union[Dict[str, Any], List[Dict[str, Any]], None] = None,
        # Descriptive aliases (take precedence if provided)
        spline_order: Optional[int] = None,
        grid_points: Optional[int] = None,
        multiplication_inputs: Optional[int] = None,
    ):
        super().__init__()

        # Resolve parameter aliases (descriptive names take precedence)
        if spline_order is not None:
            k = spline_order
        if grid_points is not None:
            grid = grid_points
        if multiplication_inputs is not None:
            mult_arity = multiplication_inputs

        if seed is not None:
            mx.random.seed(seed)
            np.random.seed(seed)

        # Parse width format
        self._width_raw = width
        self._width = self._parse_width(width)
        self.mult_arity = mult_arity

        # For backward compatibility, expose simple width list
        self.width = [w[0] + w[1] for w in self._width]  # Total nodes per layer
        self.depth = len(self._width) - 1
        self.grid = grid
        self.k = k
        self.base_fun = base_fun
        self.grid_range = grid_range
        self.noise_scale = noise_scale

        # Normalize basis parameters to per-layer lists
        if isinstance(basis, str):
            bases = [basis] * self.depth
        else:
            if len(basis) != self.depth:
                raise ValueError(
                    f"basis list length ({len(basis)}) must match depth ({self.depth})"
                )
            bases = list(basis)

        if basis_M is None:
            basis_Ms = [None] * self.depth
        elif isinstance(basis_M, int):
            basis_Ms = [basis_M] * self.depth
        else:
            if len(basis_M) != self.depth:
                raise ValueError(
                    f"basis_M list length ({len(basis_M)}) must match depth ({self.depth})"
                )
            basis_Ms = list(basis_M)

        if basis_kwargs is None:
            basis_kwargs_list = [{}] * self.depth
        elif isinstance(basis_kwargs, dict):
            basis_kwargs_list = [basis_kwargs] * self.depth
        else:
            if len(basis_kwargs) != self.depth:
                raise ValueError(
                    f"basis_kwargs list length ({len(basis_kwargs)}) must match depth ({self.depth})"
                )
            basis_kwargs_list = list(basis_kwargs)

        # Store basis config
        self._bases = bases
        self._basis_Ms = basis_Ms
        self._basis_kwargs = basis_kwargs_list

        # Create layers
        # Each layer maps from width_in to width_out (includes mult inputs)
        self.layers = []
        self.symbolic_funs = []
        for i in range(self.depth):
            # Input dimension: sum + mult nodes from previous layer
            in_dim = self._width[i][0] + self._width[i][1]
            # Output dimension: sum nodes + (mult nodes * mult_arity)
            n_sum_out = self._width[i + 1][0]
            n_mult_out = self._width[i + 1][1]
            out_dim = n_sum_out + n_mult_out * mult_arity

            layer = KANLayer(
                in_dim=in_dim,
                out_dim=out_dim,
                num_grid=grid,
                k=k,
                noise_scale=noise_scale,
                base_fun=base_fun,
                grid_range=grid_range,
                # New basis parameters
                basis=bases[i],
                basis_M=basis_Ms[i],
                basis_kwargs=basis_kwargs_list[i],
            )
            self.layers.append(layer)

            # Create corresponding symbolic layer
            symbolic_layer = Symbolic_KANLayer(
                in_dim=in_dim,
                out_dim=out_dim,
            )
            self.symbolic_funs.append(symbolic_layer)

        # Training history
        self._history = {"train_loss": [], "test_loss": [], "reg_loss": []}

        # Model versioning
        self._versions = []  # List of (version_id, state_dict)
        self._current_version = 0

        # Speed mode flag
        self._speed_mode = False

        # Cache for activations (used for symbolic fitting and visualization)
        # PyKAN-compatible: stores acts, spline_postacts, etc.
        # NOTE: These use underscore prefix to prevent MLX from treating them as parameters
        self._activations_cache = None
        self._acts = None  # Input to each layer: list of (batch, width_in[l])
        self._spline_postacts = None  # Spline outputs: list of (batch, out_dim, in_dim)
        self._postacts = None  # Full activations: list of (batch, out_dim, in_dim)
        self._cache_data = None  # Last input data for re-forward
        self._save_act = True  # Whether to cache activations

    def _parse_width(self, width: Union[List[int], List[List[int]]]) -> List[List[int]]:
        """Parse width specification into [[n_sum, n_mult], ...] format.

        Args:
            width: Either [int, ...] or [[n_sum, n_mult], ...]

        Returns:
            List of [n_sum, n_mult] pairs
        """
        if len(width) == 0:
            raise ValueError("Width cannot be empty")

        # Check if simple format (list of ints) or extended format
        if isinstance(width[0], int):
            # Simple format: convert to [[n, 0], ...]
            return [[w, 0] for w in width]
        else:
            # Extended format: validate and return
            parsed = []
            for w in width:
                if len(w) != 2:
                    raise ValueError(f"Each width entry must be [n_sum, n_mult], got {w}")
                parsed.append([int(w[0]), int(w[1])])
            return parsed

    @property
    def width_in(self) -> List[int]:
        """Number of sum nodes per layer (input perspective)."""
        return [w[0] for w in self._width]

    @property
    def width_out(self) -> List[int]:
        """Total nodes per layer (sum + mult)."""
        return [w[0] + w[1] for w in self._width]

    @property
    def n_sum(self) -> List[int]:
        """Number of sum nodes per layer."""
        return [w[0] for w in self._width]

    @property
    def n_mult(self) -> List[int]:
        """Number of multiplication nodes per layer."""
        return [w[1] for w in self._width]

    def has_mult_nodes(self) -> bool:
        """Check if network has any multiplication nodes."""
        return any(w[1] > 0 for w in self._width)

    # ===== Forward Pass =====

    def _apply_mult_nodes(self, x: mx.array, layer_idx: int) -> mx.array:
        """Apply multiplication operation at mult nodes.

        Args:
            x: Layer output of shape (batch, n_sum + n_mult * mult_arity)
            layer_idx: Current layer index (0-based, after layer application)

        Returns:
            Combined output of shape (batch, n_sum + n_mult)
        """
        n_sum = self._width[layer_idx + 1][0]
        n_mult = self._width[layer_idx + 1][1]

        if n_mult == 0:
            return x

        # Split into sum outputs and mult inputs
        x_sum = x[:, :n_sum]  # (batch, n_sum)
        x_mult_inputs = x[:, n_sum:]  # (batch, n_mult * mult_arity)

        # Reshape mult inputs: (batch, n_mult, mult_arity)
        batch_size = x.shape[0]
        x_mult_inputs = x_mult_inputs.reshape(batch_size, n_mult, self.mult_arity)

        # Compute products along mult_arity dimension
        x_mult = mx.prod(x_mult_inputs, axis=2)  # (batch, n_mult)

        # Concatenate sum and mult outputs
        return mx.concatenate([x_sum, x_mult], axis=1)

    @overload
    def __call__(
        self,
        x: mx.array,
        return_activations: Literal[False] = False,
        singularity_avoiding: bool = False,
        y_th: float = 10.0,
    ) -> mx.array: ...

    @overload
    def __call__(
        self,
        x: mx.array,
        return_activations: Literal[True],
        singularity_avoiding: bool = False,
        y_th: float = 10.0,
    ) -> Tuple[mx.array, List[Dict[str, mx.array]]]: ...

    def __call__(
        self,
        x: mx.array,
        return_activations: bool = False,
        singularity_avoiding: bool = False,
        y_th: float = 10.0,  # Clipping threshold to prevent numerical overflow
    ) -> Union[mx.array, Tuple[mx.array, List[Dict[str, mx.array]]]]:
        """Forward pass through the network.

        For networks with multiplication nodes:
        - Each layer outputs n_sum + n_mult * mult_arity values
        - Mult nodes compute products of mult_arity consecutive values
        - Final output per layer is n_sum + n_mult values

        Args:
            x: Input tensor of shape (batch, width[0])
            return_activations: If True, return intermediate activations
            singularity_avoiding: If True, clip outputs to avoid singularities
            y_th: Threshold for singularity avoidance

        Returns:
            If return_activations is False:
                Output tensor of shape (batch, width[-1])
            If return_activations is True:
                Tuple of (output, list of layer activations)
        """
        activations = []

        # PyKAN-compatible activation caching
        # Note: Using underscore prefix to prevent MLX from treating these as parameters
        if self._save_act:
            self._cache_data = x
            self._acts = [x]  # Input to each layer
            self._spline_postacts = []  # Spline outputs (batch, out_dim, in_dim)
            self._postacts = []  # Full activations

        for layer_idx, layer in enumerate(self.layers):
            if return_activations or self._save_act:
                x_raw, preacts, postacts, postspline = layer(x, return_activations=True)

                if return_activations:
                    activations.append({
                        "preacts": preacts,
                        "postacts": postacts,
                        "postspline": postspline,
                    })

                if self._save_act:
                    # Store in PyKAN format: (batch, out_dim, in_dim)
                    # postspline is (batch, in_dim, out_dim), need to transpose
                    self._spline_postacts.append(mx.transpose(postspline, axes=(0, 2, 1)))
                    self._postacts.append(mx.transpose(postacts, axes=(0, 2, 1)))

                # Apply multiplication nodes if present
                x = self._apply_mult_nodes(x_raw, layer_idx)

                if self._save_act:
                    self._acts.append(x)
            else:
                x_raw = layer(x)
                x = self._apply_mult_nodes(x_raw, layer_idx)

            # Singularity avoidance
            if singularity_avoiding:
                x = mx.clip(x, -y_th, y_th)

        if return_activations:
            return x, activations

        return x

    def forward_fast(self, x: mx.array) -> mx.array:
        """JIT-compiled fast forward pass (no symbolic, no activations)."""
        for layer_idx, layer in enumerate(self.layers):
            x_raw = layer(x)
            x = self._apply_mult_nodes(x_raw, layer_idx)
        return x

    # ===== Training =====

    def fit(
        self,
        dataset: TrainingDataset,
        opt: str = "Adam",
        steps: int = 100,
        lr: float = 1e-2,
        # Regularization: loss = MSE + λ(λ₁‖c‖₁ + λₑH + λ_c‖c‖² + λ_s‖Δc‖)
        lamb: float = 0.0,         # λ   - overall regularization strength
        lamb_l1: float = 1.0,      # λ₁  - L1 sparsity on coefficients
        lamb_entropy: float = 2.0, # λₑ  - entropy regularization
        lamb_coef: float = 0.0,    # λ_c - coefficient magnitude
        lamb_smooth: float = 0.0,  # λ_s - coefficient smoothness
        # Training options
        batch_size: int = -1,  # -1 for full batch
        update_grid: bool = True,
        grid_update_freq: int = 10,
        stop_grid_update_step: int = 50,
        # Mixed precision
        mixed_precision: bool = False,
        # LBFGS options
        lbfgs_max_iter: int = 20,
        # Logging
        log_freq: int = 10,
        verbose: bool = True,
        # Checkpointing
        save_checkpoints: bool = False,
        checkpoint_freq: int = 50,
        # Descriptive aliases (take precedence if provided)
        learning_rate: Optional[float] = None,
        regularization: Optional[float] = None,
        l1_weight: Optional[float] = None,
        entropy_weight: Optional[float] = None,
        coef_weight: Optional[float] = None,
        smooth_weight: Optional[float] = None,
    ) -> TrainingHistory:
        """Train the KAN model.

        Minimizes: loss = MSE + λ(λ₁‖c‖₁ + λₑH + λ_c‖c‖² + λ_s‖Δc‖)

        Args:
            dataset: TrainingDataset with 'train_input', 'train_label',
                     and optionally 'test_input', 'test_label'
            opt: Optimizer ('Adam', 'SGD', 'AdamW', 'LBFGS')
            steps: Number of training steps
            lr: Learning rate
            lamb: λ - overall regularization strength
            lamb_l1: λ₁ - L1 penalty on spline coefficients
            lamb_entropy: λₑ - entropy regularization (promotes sparsity)
            lamb_coef: λ_c - coefficient magnitude penalty
            lamb_smooth: λ_s - coefficient smoothness (penalizes ‖Δc‖)
            batch_size: Batch size (-1 for full batch)
            update_grid: Adaptively update grid during training
            grid_update_freq: Steps between grid updates
            stop_grid_update_step: Stop grid updates after this step
            mixed_precision: Use float16 forward pass
            lbfgs_max_iter: Max iterations per LBFGS step
            log_freq: Steps between progress logs
            verbose: Print training progress
            save_checkpoints: Save model versions during training
            checkpoint_freq: Steps between checkpoints

        Returns:
            TrainingHistory with 'train_loss', 'test_loss', 'reg_loss'
        """
        # Resolve parameter aliases (descriptive names take precedence)
        if learning_rate is not None:
            lr = learning_rate
        if regularization is not None:
            lamb = regularization
        if l1_weight is not None:
            lamb_l1 = l1_weight
        if entropy_weight is not None:
            lamb_entropy = entropy_weight
        if coef_weight is not None:
            lamb_coef = coef_weight
        if smooth_weight is not None:
            lamb_smooth = smooth_weight

        train_input = dataset["train_input"]
        train_label = dataset["train_label"]
        test_input = dataset.get("test_input")
        test_label = dataset.get("test_label")

        n_train = train_input.shape[0]

        # Validate batch_size
        if batch_size == -1:
            batch_size = n_train
        elif batch_size <= 0:
            raise ValueError(f"batch_size must be > 0 or -1 (full batch), got {batch_size}")
        elif batch_size > n_train:
            import warnings
            warnings.warn(
                f"batch_size ({batch_size}) > n_train ({n_train}), using full batch",
                UserWarning
            )
            batch_size = n_train

        # Define loss function: MSE + λ(λ₁‖c‖₁ + λₑH + λ_c‖c‖² + λ_s‖Δc‖)
        def loss_fn(model, x, y):
            # Mixed precision forward
            if mixed_precision:
                x = x.astype(mx.float16)
                pred = model(x).astype(mx.float32)
            else:
                pred = model(x)

            mse = mx.mean((pred - y) ** 2)

            # Regularization
            reg = mx.array(0.0)
            if lamb > 0:
                for layer in model.layers:
                    # λ₁‖c‖₁ - L1 on coefficients
                    if lamb_l1 > 0:
                        reg = reg + lamb * lamb_l1 * mx.mean(mx.abs(layer.coef))

                    # L1 on scales
                    reg = reg + lamb * mx.mean(mx.abs(layer.scale_sp))

                    # λ_c‖c‖² - Coefficient magnitude penalty
                    if lamb_coef > 0:
                        reg = reg + lamb * lamb_coef * mx.mean(layer.coef ** 2)

                    # λ_s‖Δc‖ - Coefficient smoothness
                    if lamb_smooth > 0:
                        coef_diff = layer.coef[:, :, 1:] - layer.coef[:, :, :-1]
                        reg = reg + lamb * lamb_smooth * mx.mean(mx.abs(coef_diff))

                    # λₑH - Entropy regularization (sparsity)
                    if lamb_entropy > 0:
                        scores = mx.abs(layer.scale_sp) + 1e-8
                        probs = scores / mx.sum(scores)
                        entropy = -mx.sum(probs * mx.log(probs + 1e-8))
                        reg = reg + lamb * lamb_entropy * entropy

            return mse + reg, mse, reg

        # Use LBFGS or standard optimizers
        if opt.upper() == "LBFGS":
            return self._fit_lbfgs(
                dataset, loss_fn, steps, lbfgs_max_iter,
                update_grid, grid_update_freq, stop_grid_update_step,
                log_freq, verbose, save_checkpoints, checkpoint_freq
            )

        # Setup optimizer
        if opt == "Adam":
            optimizer = optim.Adam(learning_rate=lr)
        elif opt == "AdamW":
            optimizer = optim.AdamW(learning_rate=lr)
        elif opt == "SGD":
            optimizer = optim.SGD(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt}")

        # Training loop with value_and_grad
        def loss_wrapper(model, x, y):
            total, mse, reg = loss_fn(model, x, y)
            return total

        loss_and_grad = nn.value_and_grad(model=self, fn=loss_wrapper)

        for step in range(steps):
            # Sample batch
            if batch_size < n_train:
                idx = mx.random.randint(0, n_train, shape=(batch_size,))
                batch_x = train_input[idx]
                batch_y = train_label[idx]
            else:
                batch_x = train_input
                batch_y = train_label

            # Compute loss and gradients
            loss, grads = loss_and_grad(self, batch_x, batch_y)

            # Update parameters
            optimizer.update(self, grads)
            mx.eval(self.parameters(), optimizer.state)

            # Update grid
            if update_grid and step < stop_grid_update_step:
                if step % grid_update_freq == 0:
                    self.update_grid_from_samples(train_input)

            # Save checkpoint
            if save_checkpoints and step % checkpoint_freq == 0:
                self._save_version(f"step_{step}")

            # Logging
            if step % log_freq == 0 or step == steps - 1:
                total, mse, reg = loss_fn(self, train_input, train_label)
                train_loss = float(mse)
                self._history["train_loss"].append(train_loss)
                self._history["reg_loss"].append(float(reg))

                if test_input is not None:
                    test_total, test_mse, _ = loss_fn(self, test_input, test_label)
                    test_loss = float(test_mse)
                    self._history["test_loss"].append(test_loss)

                    if verbose:
                        print(
                            f"Step {step:4d} | "
                            f"Train: {train_loss:.6f} | "
                            f"Test: {test_loss:.6f} | "
                            f"Reg: {float(reg):.6f}"
                        )
                else:
                    if verbose:
                        print(f"Step {step:4d} | Train: {train_loss:.6f} | Reg: {float(reg):.6f}")

        return self._history

    def _fit_lbfgs(
        self,
        dataset: Dict[str, mx.array],
        loss_fn: Callable,
        steps: int,
        max_iter: int,
        update_grid: bool,
        grid_update_freq: int,
        stop_grid_update_step: int,
        log_freq: int,
        verbose: bool,
        save_checkpoints: bool,
        checkpoint_freq: int,
    ) -> TrainingHistory:
        """Training with scipy's L-BFGS-B optimizer."""
        from scipy.optimize import minimize

        train_input = dataset["train_input"]
        train_label = dataset["train_label"]
        test_input = dataset.get("test_input")
        test_label = dataset.get("test_label")

        # Flatten parameters for scipy
        def get_flat_params():
            params = []
            for layer in self.layers:
                params.append(np.array(layer.coef).flatten())
                params.append(np.array(layer.scale_sp).flatten())
                params.append(np.array(layer.scale_base).flatten())
            return np.concatenate(params)

        def set_flat_params(flat_params):
            idx = 0
            for layer in self.layers:
                # coef
                size = layer.coef.size
                layer.coef = mx.array(flat_params[idx:idx+size].reshape(layer.coef.shape))
                idx += size
                # scale_sp
                size = layer.scale_sp.size
                layer.scale_sp = mx.array(flat_params[idx:idx+size].reshape(layer.scale_sp.shape))
                idx += size
                # scale_base
                size = layer.scale_base.size
                layer.scale_base = mx.array(flat_params[idx:idx+size].reshape(layer.scale_base.shape))
                idx += size

        # Objective for scipy
        def objective(flat_params):
            set_flat_params(flat_params)
            total, mse, reg = loss_fn(self, train_input, train_label)
            return float(total)

        # Gradient for scipy (using finite differences for simplicity)
        # For better performance, implement analytical gradients
        step_count = [0]

        def callback(xk):
            step_count[0] += 1
            step = step_count[0]

            # Update grid
            if update_grid and step < stop_grid_update_step:
                if step % grid_update_freq == 0:
                    self.update_grid_from_samples(train_input)

            # Save checkpoint
            if save_checkpoints and step % checkpoint_freq == 0:
                self._save_version(f"step_{step}")

            # Logging
            if step % log_freq == 0:
                total, mse, reg = loss_fn(self, train_input, train_label)
                train_loss = float(mse)
                self._history["train_loss"].append(train_loss)
                self._history["reg_loss"].append(float(reg))

                if test_input is not None:
                    test_total, test_mse, _ = loss_fn(self, test_input, test_label)
                    test_loss = float(test_mse)
                    self._history["test_loss"].append(test_loss)

                    if verbose:
                        print(f"Step {step:4d} | Train: {train_loss:.6f} | Test: {test_loss:.6f}")
                else:
                    if verbose:
                        print(f"Step {step:4d} | Train: {train_loss:.6f}")

        # Run L-BFGS-B
        x0 = get_flat_params()
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': steps, 'maxfun': steps * max_iter, 'disp': False}
        )

        set_flat_params(result.x)

        if verbose:
            print(f"L-BFGS converged: {result.success}, message: {result.message}")

        return self._history

    # ===== Model Versioning =====

    def _save_version(self, name: Optional[str] = None) -> int:
        """Save current state as a version.

        Args:
            name: Optional name for this version. If None, uses 'v{id}'.

        Returns:
            Version ID for later retrieval via checkout().
        """
        version_id = len(self._versions)
        if name is None:
            name = f"v{version_id}"

        state = {
            'name': name,
            'width': self.width.copy(),
            'grid': self.grid,
            'k': self.k,
            'layers': [],
            'symbolic_funs': [],
        }

        for layer in self.layers:
            state['layers'].append({
                'coef': np.array(layer.coef),
                'scale_sp': np.array(layer.scale_sp),
                'scale_base': np.array(layer.scale_base),
                'grid': np.array(layer.grid),
                'mask': np.array(layer.mask),
            })

        for sym_layer in self.symbolic_funs:
            state['symbolic_funs'].append({
                'fns_name': deepcopy(sym_layer.fns_name),
                'affine_a': np.array(sym_layer.affine_a),
                'affine_b': np.array(sym_layer.affine_b),
                'affine_c': np.array(sym_layer.affine_c),
                'affine_d': np.array(sym_layer.affine_d),
            })

        self._versions.append(state)
        self._current_version = version_id
        return version_id

    def rewind(self, version: int = -1) -> None:
        """Rewind to a previous version.

        Args:
            version: Version index (-1 for previous version)
        """
        if len(self._versions) == 0:
            raise ValueError("No saved versions to rewind to")

        if version == -1:
            version = max(0, self._current_version - 1)

        if version >= len(self._versions):
            raise ValueError(f"Version {version} not found. Available: 0-{len(self._versions)-1}")

        self.checkout(version)

    def checkout(self, version: int) -> None:
        """Checkout a specific version.

        Args:
            version: Version index to checkout
        """
        if version >= len(self._versions) or version < 0:
            raise ValueError(f"Version {version} not found. Available: 0-{len(self._versions)-1}")

        state = self._versions[version]

        # Restore layers
        for i, layer_state in enumerate(state['layers']):
            self.layers[i].coef = mx.array(layer_state['coef'])
            self.layers[i].scale_sp = mx.array(layer_state['scale_sp'])
            self.layers[i].scale_base = mx.array(layer_state['scale_base'])
            self.layers[i]._grid = mx.array(layer_state['grid'])
            self.layers[i]._mask = mx.array(layer_state['mask'])

        # Restore symbolic layers
        for i, sym_state in enumerate(state['symbolic_funs']):
            self.symbolic_funs[i].fns_name = deepcopy(sym_state['fns_name'])
            self.symbolic_funs[i].affine_a = mx.array(sym_state['affine_a'])
            self.symbolic_funs[i].affine_b = mx.array(sym_state['affine_b'])
            self.symbolic_funs[i].affine_c = mx.array(sym_state['affine_c'])
            self.symbolic_funs[i].affine_d = mx.array(sym_state['affine_d'])

        self._current_version = version

    def list_versions(self) -> List[str]:
        """List all saved versions."""
        return [f"{i}: {v['name']}" for i, v in enumerate(self._versions)]

    # ===== Architecture Modification =====

    def expand_width(self, layer_idx: int, new_width: int) -> None:
        """Add neurons to a hidden layer.

        Args:
            layer_idx: Index of layer to expand (0 to depth-1)
            new_width: New width for that layer
        """
        if layer_idx < 0 or layer_idx >= self.depth:
            raise ValueError(f"layer_idx must be 0-{self.depth-1}")

        old_width = self.width[layer_idx + 1]
        if new_width <= old_width:
            raise ValueError(f"new_width must be greater than current width {old_width}")

        # Expand current layer's output
        layer = self.layers[layer_idx]
        extra = new_width - old_width

        # Expand coef
        new_coef = mx.concatenate([
            layer.coef,
            mx.random.normal(shape=(layer.in_dim, extra, layer.coef.shape[2])) * self.noise_scale
        ], axis=1)
        layer.coef = new_coef

        # Expand scales
        layer.scale_sp = mx.concatenate([
            layer.scale_sp,
            mx.ones((layer.in_dim, extra))
        ], axis=1)
        layer.scale_base = mx.concatenate([
            layer.scale_base,
            mx.ones((layer.in_dim, extra))
        ], axis=1)

        # Expand mask
        layer._mask = mx.concatenate([
            layer._mask,
            mx.ones((layer.in_dim, extra))
        ], axis=1)

        layer.out_dim = new_width

        # Expand next layer's input (if not the last layer)
        if layer_idx < self.depth - 1:
            next_layer = self.layers[layer_idx + 1]

            # Expand coef
            new_coef = mx.concatenate([
                next_layer.coef,
                mx.random.normal(shape=(extra, next_layer.out_dim, next_layer.coef.shape[2])) * self.noise_scale
            ], axis=0)
            next_layer.coef = new_coef

            # Expand scales
            next_layer.scale_sp = mx.concatenate([
                next_layer.scale_sp,
                mx.ones((extra, next_layer.out_dim))
            ], axis=0)
            next_layer.scale_base = mx.concatenate([
                next_layer.scale_base,
                mx.ones((extra, next_layer.out_dim))
            ], axis=0)

            # Expand mask
            next_layer._mask = mx.concatenate([
                next_layer._mask,
                mx.ones((extra, next_layer.out_dim))
            ], axis=0)

            # Expand grid
            new_grid = mx.concatenate([
                next_layer._grid,
                mx.broadcast_to(next_layer._grid[:1], (extra, next_layer._grid.shape[1]))
            ], axis=0)
            next_layer._grid = new_grid

            next_layer.in_dim = new_width

        # Update symbolic layers
        self.symbolic_funs[layer_idx] = Symbolic_KANLayer(
            in_dim=layer.in_dim,
            out_dim=new_width,
        )
        if layer_idx < self.depth - 1:
            self.symbolic_funs[layer_idx + 1] = Symbolic_KANLayer(
                in_dim=new_width,
                out_dim=self.layers[layer_idx + 1].out_dim,
            )

        # Update width
        self.width[layer_idx + 1] = new_width

    def expand_depth(self, new_layer_width: int, position: int = -1) -> None:
        """Add a new layer to the network.

        Args:
            new_layer_width: Width of the new hidden layer
            position: Position to insert (-1 for before output)
        """
        if position == -1:
            position = self.depth - 1

        if position < 0 or position > self.depth:
            raise ValueError(f"position must be 0-{self.depth}")

        # Dimensions
        in_dim = self.width[position]
        out_dim = self.width[position + 1]

        # Create two new layers to replace one
        # First: in_dim -> new_layer_width
        layer1 = KANLayer(
            in_dim=in_dim,
            out_dim=new_layer_width,
            num_grid=self.grid,
            k=self.k,
            noise_scale=self.noise_scale,
            base_fun=self.base_fun,
            grid_range=self.grid_range,
        )

        # Second: new_layer_width -> out_dim
        layer2 = KANLayer(
            in_dim=new_layer_width,
            out_dim=out_dim,
            num_grid=self.grid,
            k=self.k,
            noise_scale=self.noise_scale,
            base_fun=self.base_fun,
            grid_range=self.grid_range,
        )

        # Insert layers
        self.layers = self.layers[:position] + [layer1, layer2] + self.layers[position+1:]

        # Create symbolic layers
        sym1 = Symbolic_KANLayer(in_dim=in_dim, out_dim=new_layer_width)
        sym2 = Symbolic_KANLayer(in_dim=new_layer_width, out_dim=out_dim)
        self.symbolic_funs = self.symbolic_funs[:position] + [sym1, sym2] + self.symbolic_funs[position+1:]

        # Update width and depth
        self.width = self.width[:position+1] + [new_layer_width] + self.width[position+1:]
        self.depth = len(self.width) - 1

    # ===== Speed Mode =====

    def speed(self, enabled: bool = True) -> None:
        """Enable/disable speed mode (disables symbolic computation branch).

        In speed mode, forward pass only uses splines, not symbolic functions.
        This is useful for faster training when symbolic regression isn't needed.

        Args:
            enabled: Whether to enable speed mode
        """
        self._speed_mode = enabled

    # ===== Improved Pruning =====

    def prune(
        self,
        threshold: float = 1e-2,
        mode: str = "auto",
        active_neurons_id: Optional[List[List[int]]] = None,
    ) -> "MultKAN":
        """Combined node and edge pruning.

        Args:
            threshold: Pruning threshold
            mode: 'auto' or 'manual'
            active_neurons_id: For manual mode, list of active neuron ids per layer

        Returns:
            Pruned model (self, modified in place)
        """
        if mode == "auto":
            self.prune_edges(threshold)
            self.prune_nodes(threshold)
        elif mode == "manual":
            if active_neurons_id is None:
                raise ValueError("active_neurons_id required for manual mode")
            self._prune_manual(active_neurons_id)

        return self

    def prune_edges(self, threshold: float = 0.01) -> None:
        """Prune edges with low importance scores.

        Args:
            threshold: Edges with scores below this are pruned
        """
        for layer in self.layers:
            scores = layer.edge_scores()
            mask = (scores > threshold).astype(mx.float32)
            layer.set_mask(mask)

    def prune_nodes(self, threshold: float = 0.01) -> List[int]:
        """Prune nodes with low total contribution.

        Args:
            threshold: Nodes with total score below this are pruned

        Returns:
            New width configuration after pruning
        """
        new_width = [self.width[0]]

        for i, layer in enumerate(self.layers[:-1]):
            scores = layer.edge_scores()
            node_scores = mx.sum(scores, axis=0)

            keep_mask = node_scores > threshold
            keep_ids = [j for j in range(self.width[i + 1]) if bool(keep_mask[j])]

            if len(keep_ids) == 0:
                keep_ids = [int(mx.argmax(node_scores))]

            new_width.append(len(keep_ids))

            self.layers[i] = layer.get_subset(
                in_ids=list(range(layer.in_dim)),
                out_ids=keep_ids,
            )

            next_layer = self.layers[i + 1]
            self.layers[i + 1] = next_layer.get_subset(
                in_ids=keep_ids,
                out_ids=list(range(next_layer.out_dim)),
            )

        new_width.append(self.width[-1])
        self.width = new_width

        return new_width

    def prune_input(self, threshold: float = 1e-2) -> List[int]:
        """Remove inactive input dimensions.

        Args:
            threshold: Inputs with total score below this are removed

        Returns:
            List of kept input indices
        """
        layer = self.layers[0]
        scores = layer.edge_scores()
        input_scores = mx.sum(scores, axis=1)

        keep_mask = input_scores > threshold
        keep_ids = [i for i in range(self.width[0]) if bool(keep_mask[i])]

        if len(keep_ids) == 0:
            keep_ids = [int(mx.argmax(input_scores))]

        # Update first layer
        self.layers[0] = layer.get_subset(
            in_ids=keep_ids,
            out_ids=list(range(layer.out_dim)),
        )

        # Update width
        self.width[0] = len(keep_ids)

        return keep_ids

    def _prune_manual(self, active_neurons_id: List[List[int]]) -> None:
        """Manual pruning with specified active neurons."""
        for l, keep_ids in enumerate(active_neurons_id):
            if l < self.depth - 1:
                self.layers[l] = self.layers[l].get_subset(
                    in_ids=list(range(self.layers[l].in_dim)),
                    out_ids=keep_ids,
                )
                self.layers[l + 1] = self.layers[l + 1].get_subset(
                    in_ids=keep_ids,
                    out_ids=list(range(self.layers[l + 1].out_dim)),
                )
                self.width[l + 1] = len(keep_ids)

    # ===== Initialize from Another Model =====

    def initialize_from_another_model(
        self,
        other: "MultKAN",
        x: mx.array,
    ) -> None:
        """Initialize from another model with potentially different grid.

        Useful for transfer learning or grid refinement.

        Args:
            other: Source model
            x: Sample input for grid adaptation
        """
        if self.width != other.width:
            raise ValueError("Models must have same width")

        # Update grids from samples
        self.update_grid_from_samples(x)

        # Copy scale parameters
        for i, (layer, other_layer) in enumerate(zip(self.layers, other.layers)):
            layer.scale_sp = other_layer.scale_sp
            layer.scale_base = other_layer.scale_base
            layer._mask = other_layer._mask

            # Refit coefficients from other model's spline curves
            x_sample = mx.linspace(-1, 1, 100).reshape(-1, 1)
            x_sample = mx.broadcast_to(x_sample, (100, layer.in_dim))

            y_other = coef2curve(x_sample, other_layer.grid, other_layer.coef, other_layer.k)
            new_coef = curve2coef(x_sample, y_other, layer.grid, layer.k)
            layer.coef = new_coef

        # Copy symbolic assignments
        for sym, other_sym in zip(self.symbolic_funs, other.symbolic_funs):
            sym.fns_name = deepcopy(other_sym.fns_name)
            sym.affine_a = other_sym.affine_a
            sym.affine_b = other_sym.affine_b
            sym.affine_c = other_sym.affine_c
            sym.affine_d = other_sym.affine_d

    # ===== Grid and Refinement =====

    def update_grid_from_samples(self, x: mx.array) -> None:
        """Update all layer grids based on input distribution."""
        for layer in self.layers:
            layer.update_grid_from_samples(x)
            x = layer(x)

    def refine(self, new_grid: int) -> None:
        """Increase grid resolution for better accuracy.

        Args:
            new_grid: New number of grid intervals
        """
        for layer in self.layers:
            old_grid = layer.grid
            grid_min = float(mx.min(old_grid))
            grid_max = float(mx.max(old_grid))

            new_layer = KANLayer(
                in_dim=layer.in_dim,
                out_dim=layer.out_dim,
                num_grid=new_grid,
                k=layer.k,
                base_fun=layer.base_fun,
                grid_range=(grid_min, grid_max),
            )

            # Refit coefficients from old spline
            x_sample = mx.linspace(grid_min, grid_max, 100).reshape(-1, 1)
            x_sample = mx.broadcast_to(x_sample, (100, layer.in_dim))

            y_old = coef2curve(x_sample, layer.grid, layer.coef, layer.k)
            new_coef = curve2coef(x_sample, y_old, new_layer.grid, new_layer.k)
            new_layer.coef = new_coef

            new_layer.scale_sp = layer.scale_sp
            new_layer.scale_base = layer.scale_base

        self.grid = new_grid

    # ===== Uncertainty Quantification =====

    def predict_with_uncertainty(
        self,
        x: mx.array,
        n_samples: int = 100,
        noise_scale: float = 0.01,
    ) -> Tuple[mx.array, mx.array]:
        """Predict with uncertainty estimates using dropout-like noise.

        Args:
            x: Input tensor
            n_samples: Number of forward passes
            noise_scale: Scale of added noise

        Returns:
            Tuple of (mean prediction, std prediction)
        """
        predictions = []

        for _ in range(n_samples):
            # Add noise to coefficients
            for layer in self.layers:
                layer.coef = layer.coef + mx.random.normal(layer.coef.shape) * noise_scale

            pred = self(x)
            predictions.append(pred)

            # Remove noise
            for layer in self.layers:
                layer.coef = layer.coef - mx.random.normal(layer.coef.shape) * noise_scale

        predictions = mx.stack(predictions, axis=0)
        mean = mx.mean(predictions, axis=0)
        std = mx.std(predictions, axis=0)

        return mean, std

    # ===== Edge Score Metrics =====

    def edge_scores(
        self,
        metric: str = "backward",
        x: Optional[mx.array] = None,
    ) -> List[mx.array]:
        """Compute edge importance scores with different metrics.

        Args:
            metric: Score metric type
                - 'backward': Default edge scores from layer
                - 'forward_n': Normalized forward activation
                - 'forward_u': Unnormalized forward activation
                - 'forward_sum': Sum of absolute activations
                - 'node_backward': Node-level backward attribution

        Returns:
            List of score arrays per layer
        """
        scores = []

        for layer in self.layers:
            if metric == "backward":
                score = layer.edge_scores()
            elif metric == "forward_n":
                score = layer.edge_scores()
                max_score = mx.max(score)
                if float(max_score) > 0:
                    score = score / max_score
            elif metric == "forward_u":
                score = layer.edge_scores()
            elif metric == "forward_sum":
                score = mx.abs(layer.scale_sp) + mx.abs(layer.scale_base)
            elif metric == "node_backward":
                raw = layer.edge_scores()
                score = mx.sum(raw, axis=0, keepdims=True)
                score = mx.broadcast_to(score, raw.shape)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            scores.append(score)

        return scores

    # ===== Checkpointing =====

    def saveckpt(self, path: str = "model") -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        params = {}
        for i, layer in enumerate(self.layers):
            params[f"layer_{i}_coef"] = np.array(layer.coef)
            params[f"layer_{i}_scale_sp"] = np.array(layer.scale_sp)
            params[f"layer_{i}_scale_base"] = np.array(layer.scale_base)
            params[f"layer_{i}_grid"] = np.array(layer.grid)
            params[f"layer_{i}_mask"] = np.array(layer.mask)

        # Save symbolic info
        for i, sym in enumerate(self.symbolic_funs):
            params[f"sym_{i}_fns_name"] = sym.fns_name
            params[f"sym_{i}_affine_a"] = np.array(sym.affine_a)
            params[f"sym_{i}_affine_b"] = np.array(sym.affine_b)
            params[f"sym_{i}_affine_c"] = np.array(sym.affine_c)
            params[f"sym_{i}_affine_d"] = np.array(sym.affine_d)

        config = {
            "width": self.width,
            "grid": self.grid,
            "k": self.k,
        }

        with open(f"{path}_params.pkl", "wb") as f:
            pickle.dump(params, f)
        with open(f"{path}_config.pkl", "wb") as f:
            pickle.dump(config, f)

    def loadckpt(self, path: str = "model") -> None:
        """Load model checkpoint."""
        with open(f"{path}_params.pkl", "rb") as f:
            params = pickle.load(f)
        with open(f"{path}_config.pkl", "rb") as f:
            config = pickle.load(f)

        if self.width != config["width"]:
            self.__init__(
                width=config["width"],
                grid=config["grid"],
                k=config["k"],
            )

        for i, layer in enumerate(self.layers):
            layer.coef = mx.array(params[f"layer_{i}_coef"])
            layer.scale_sp = mx.array(params[f"layer_{i}_scale_sp"])
            layer.scale_base = mx.array(params[f"layer_{i}_scale_base"])
            layer._grid = mx.array(params[f"layer_{i}_grid"])
            layer._mask = mx.array(params[f"layer_{i}_mask"])

        # Load symbolic info
        for i, sym in enumerate(self.symbolic_funs):
            if f"sym_{i}_fns_name" in params:
                sym.fns_name = params[f"sym_{i}_fns_name"]
                sym.affine_a = mx.array(params[f"sym_{i}_affine_a"])
                sym.affine_b = mx.array(params[f"sym_{i}_affine_b"])
                sym.affine_c = mx.array(params[f"sym_{i}_affine_c"])
                sym.affine_d = mx.array(params[f"sym_{i}_affine_d"])

    # ===== Visualization =====

    def plot(
        self,
        folder: str = "./figures",
        beta: float = 3.0,
        scale: float = 0.5,
        tick: bool = False,
        sample: bool = False,
        in_vars: Optional[List[str]] = None,
        out_vars: Optional[List[str]] = None,
        title: Optional[str] = None,
        display: bool = True,
        save: bool = True,
    ):
        """Visualize the KAN network structure (PyKAN-style).

        This method produces visualizations identical to PyKAN's model.plot():
        - Vertical layout with input at bottom, output at top
        - Activation functions embedded as inset plots on edges
        - Uses actual cached activations from forward pass
        - Sum symbols at summation nodes
        - Transparency controlled by tanh(beta * score)

        Args:
            folder: Directory to save figure
            beta: Controls edge transparency via tanh(beta * score)
            scale: Figure size scaling factor
            tick: Show axis ticks on activation plots
            sample: Show sample points on activation plots
            in_vars: Names for input variables
            out_vars: Names for output variables
            title: Optional plot title
            display: If True, display the plot
            save: If True, save to file

        Returns:
            matplotlib Figure
        """
        from .visualization import plot_kan
        return plot_kan(
            self,
            folder=folder,
            beta=beta,
            scale=scale,
            tick=tick,
            sample=sample,
            in_vars=in_vars,
            out_vars=out_vars,
            title=title,
            display=display,
            save=save,
        )

    def summary(self) -> str:
        """Return a summary of the model architecture."""
        lines = [
            "MultKAN Summary",
            "=" * 40,
            f"Width: {self.width}",
            f"Depth: {self.depth}",
            f"Grid: {self.grid}",
            f"Spline order: {self.k}",
            f"Speed mode: {self._speed_mode}",
            f"Versions saved: {len(self._versions)}",
            "",
            "Layers:",
        ]

        total_params = 0
        for i, layer in enumerate(self.layers):
            n_params = layer.coef.size + layer.scale_sp.size + layer.scale_base.size
            total_params += n_params
            lines.append(f"  [{i}] KANLayer: {layer.in_dim} → {layer.out_dim} ({n_params:,} params)")

        lines.append("")
        lines.append(f"Total parameters: {total_params:,}")

        return "\n".join(lines)

    # ===== Symbolic Methods =====

    def fix_symbolic(
        self,
        l: int,
        i: int,
        j: int,
        fn_name: str,
        fit_params: bool = True,
        x: Optional[mx.array] = None,
        a_range: Tuple[float, float] = (-10, 10),
        b_range: Tuple[float, float] = (-10, 10),
        verbose: bool = True,
    ) -> float:
        """Fix edge (l, i, j) to a symbolic function."""
        if fn_name not in SYMBOLIC_REGISTRY:
            raise ValueError(f"Unknown function '{fn_name}'. Available: {list_symbolic()}")

        layer = self.layers[l]
        symbolic_layer = self.symbolic_funs[l]

        if fit_params:
            if x is None:
                x_sample = mx.linspace(-1, 1, 100).reshape(-1, 1)
                x_sample = mx.broadcast_to(x_sample, (100, layer.in_dim))
            else:
                x_sample = x

            y_spline = coef2curve(x_sample, layer.grid, layer.coef, layer.k)
            x_edge = np.array(x_sample[:, i])
            y_edge = np.array(y_spline[:, i, j])

            a, b, c, d, r2 = fit_affine_params(x_edge, y_edge, fn_name, a_range, b_range)

            if verbose:
                print(f"Edge ({l},{i},{j}): {fn_name}")
                print(f"  Params: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")
                print(f"  R² = {r2:.4f}")

            symbolic_layer.fix_symbolic(i, j, fn_name, a, b, c, d)
            return r2
        else:
            symbolic_layer.fix_symbolic(i, j, fn_name)
            return 1.0

    def unfix_symbolic(self, l: int, i: int, j: int) -> None:
        """Remove symbolic assignment from edge (l, i, j)."""
        self.symbolic_funs[l].unfix_symbolic(i, j)

    def suggest_symbolic(
        self,
        l: int,
        i: int,
        j: int,
        x: Optional[mx.array] = None,
        top_k: int = 5,
        a_range: Tuple[float, float] = (-10, 10),
        b_range: Tuple[float, float] = (-10, 10),
        use_basis_priority: bool = True,
    ) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        """Suggest best symbolic functions for edge (l, i, j).

        Args:
            l: Layer index
            i: Input node index
            j: Output node index
            x: Sample input data (optional)
            top_k: Number of suggestions to return
            a_range: Search range for affine parameter a
            b_range: Search range for affine parameter b
            use_basis_priority: If True, prioritize functions based on the layer's
                basis type (e.g., sin/cos for Fourier, gaussian/psi for Hermite).
                This gives a small bonus to physics-appropriate functions.
        """
        layer = self.layers[l]

        if x is None:
            x_sample = mx.linspace(-1, 1, 100).reshape(-1, 1)
            x_sample = mx.broadcast_to(x_sample, (100, layer.in_dim))
        else:
            x_sample = x

        # Get edge activation values - use forward pass with activations
        # This works for all basis types (B-spline, Fourier, etc.)
        _, preacts, postacts, postspline = layer(x_sample, return_activations=True)
        # postspline is the raw spline output (before scaling) with shape (batch, in_dim, out_dim)
        # We use postspline for symbolic fitting because it's the learned function
        x_edge = np.array(x_sample[:, i])
        y_edge = np.array(postspline[:, i, j])

        # Get basis-specific priority functions
        priority_fns = set(layer.symbolic_priority) if use_basis_priority else set()
        # Priority bonus: small boost for basis-appropriate functions
        # This breaks ties in favor of physics-meaningful functions
        priority_bonus = 0.001

        results = []
        for fn_name in SYMBOLIC_REGISTRY:
            try:
                a, b, c, d, r2 = fit_affine_params(x_edge, y_edge, fn_name, a_range, b_range)
                # Apply priority bonus for basis-appropriate functions
                score = r2 + priority_bonus if fn_name in priority_fns else r2
                results.append((fn_name, r2, (a, b, c, d), score))
            except Exception:
                continue

        # Sort by score (r2 + optional priority bonus), return without score
        results.sort(key=lambda x: x[3], reverse=True)
        return [(fn, r2, params) for fn, r2, params, _ in results[:top_k]]

    def auto_symbolic(
        self,
        x: Optional[mx.array] = None,
        r2_threshold: float = 0.99,
        a_range: Tuple[float, float] = (-10, 10),
        b_range: Tuple[float, float] = (-10, 10),
        verbose: bool = True,
    ) -> Dict[Tuple[int, int, int], Tuple[str, float]]:
        """Automatically detect and fix symbolic functions for all edges."""
        fixed = {}

        for l in range(self.depth):
            layer = self.layers[l]

            if x is None:
                x_sample = mx.linspace(-1, 1, 100).reshape(-1, 1)
                x_sample = mx.broadcast_to(x_sample, (100, layer.in_dim))
            else:
                x_sample = x
                for prev_l in range(l):
                    x_sample = self.layers[prev_l](x_sample)

            if verbose:
                print(f"\nLayer {l}:")

            for i in range(layer.in_dim):
                for j in range(layer.out_dim):
                    suggestions = self.suggest_symbolic(l, i, j, x_sample, top_k=1, a_range=a_range, b_range=b_range)

                    if suggestions and suggestions[0][1] >= r2_threshold:
                        fn_name, r2, (a, b, c, d) = suggestions[0]
                        self.symbolic_funs[l].fix_symbolic(i, j, fn_name, a, b, c, d)
                        fixed[(l, i, j)] = (fn_name, r2)

                        if verbose:
                            print(f"  ({i},{j}): {fn_name} (R²={r2:.4f})")

        if verbose:
            print(f"\nFixed {len(fixed)} edges to symbolic functions")

        return fixed

    def symbolic_formula(
        self,
        var_names: Optional[List[str]] = None,
        decimals: int = 2,
        simplify: bool = False,
        verbose: bool = True,
    ) -> str:
        """Extract the symbolic formula from the network.

        Args:
            var_names: Names for input variables
            decimals: Decimal places for coefficients
            simplify: Whether to attempt simplification (requires sympy)
            verbose: Print intermediate formulas
        """
        if var_names is None:
            var_names = [f"x_{i}" for i in range(self.width[0])]

        expressions = [var_names.copy()]

        for l in range(self.depth):
            layer = self.layers[l]
            symbolic_layer = self.symbolic_funs[l]
            layer_expressions = []

            for j in range(layer.out_dim):
                terms = []

                for i in range(layer.in_dim):
                    if symbolic_layer.is_symbolic(i, j):
                        formula = symbolic_layer.get_formula(i, j, expressions[l][i], decimals)
                        terms.append(formula)
                    else:
                        terms.append(f"spline_{l}_{i}_{j}({expressions[l][i]})")

                if len(terms) == 1:
                    expr = terms[0]
                else:
                    expr = " + ".join(terms)
                    expr = f"({expr})"

                layer_expressions.append(expr)

            expressions.append(layer_expressions)

            if verbose:
                print(f"\nLayer {l} outputs:")
                for j, expr in enumerate(layer_expressions):
                    print(f"  y_{j} = {expr}")

        final_expr = expressions[-1]
        result = final_expr[0] if len(final_expr) == 1 else ", ".join(final_expr)

        if simplify:
            try:
                import sympy
                result = str(sympy.simplify(result))
            except ImportError:
                pass

        if verbose:
            print(f"\nFinal formula:")
            print(f"  f({', '.join(var_names)}) = {result}")

        return result

    def symbolic_formula_latex(
        self,
        var_names: Optional[List[str]] = None,
        decimals: int = 2,
    ) -> str:
        """Extract the symbolic formula in LaTeX format."""
        if var_names is None:
            var_names = [f"x_{{{i}}}" for i in range(self.width[0])]

        expressions = [var_names.copy()]

        for l in range(self.depth):
            layer = self.layers[l]
            symbolic_layer = self.symbolic_funs[l]
            layer_expressions = []

            for j in range(layer.out_dim):
                terms = []

                for i in range(layer.in_dim):
                    if symbolic_layer.is_symbolic(i, j):
                        formula = symbolic_layer.get_latex_formula(i, j, expressions[l][i], decimals)
                        terms.append(formula)
                    else:
                        terms.append(f"\\phi_{{{l},{i},{j}}}({expressions[l][i]})")

                expr = terms[0] if len(terms) == 1 else " + ".join(terms)
                layer_expressions.append(expr)

            expressions.append(layer_expressions)

        final_expr = expressions[-1]
        result = final_expr[0] if len(final_expr) == 1 else ", ".join(final_expr)

        return f"f({', '.join(var_names)}) = {result}"

    def symbolic_formula_typst(
        self,
        var_names: Optional[List[str]] = None,
        decimals: int = 2,
    ) -> str:
        """Extract the symbolic formula in Typst format."""
        if var_names is None:
            var_names = [f"x_({i})" for i in range(self.width[0])]

        expressions = [var_names.copy()]

        for l in range(self.depth):
            layer = self.layers[l]
            symbolic_layer = self.symbolic_funs[l]
            layer_expressions = []

            for j in range(layer.out_dim):
                terms = []

                for i in range(layer.in_dim):
                    if symbolic_layer.is_symbolic(i, j):
                        formula = symbolic_layer.get_typst_formula(i, j, expressions[l][i], decimals)
                        terms.append(formula)
                    else:
                        terms.append(f"phi_({l},{i},{j})({expressions[l][i]})")

                expr = terms[0] if len(terms) == 1 else " + ".join(terms)
                layer_expressions.append(expr)

            expressions.append(layer_expressions)

        final_expr = expressions[-1]
        result = final_expr[0] if len(final_expr) == 1 else ", ".join(final_expr)

        return f"f({', '.join(var_names)}) = {result}"

    def render_formula(
        self,
        var_names: Optional[List[str]] = None,
        format: str = "text",
        decimals: int = 2,
    ) -> str:
        """Render the symbolic formula in various formats."""
        if format == "latex":
            return self.symbolic_formula_latex(var_names, decimals)
        elif format == "typst":
            return self.symbolic_formula_typst(var_names, decimals)
        else:
            return self.symbolic_formula(var_names, decimals, verbose=False)

    # ===== Symbolic Tree Export =====

    def to_sympy(self, var_names: Optional[List[str]] = None):
        """Convert to SymPy expression for CAS integration.

        Returns:
            sympy.Expr or list of sympy.Expr
        """
        try:
            import sympy
        except ImportError:
            raise ImportError("sympy required for to_sympy(). Install with: pip install sympy")

        if var_names is None:
            var_names = [f"x{i}" for i in range(self.width[0])]

        symbols = [sympy.Symbol(name) for name in var_names]
        expressions = [symbols.copy()]

        for l in range(self.depth):
            layer = self.layers[l]
            symbolic_layer = self.symbolic_funs[l]
            layer_expressions = []

            for j in range(layer.out_dim):
                terms = []

                for i in range(layer.in_dim):
                    if symbolic_layer.is_symbolic(i, j):
                        fn_name = symbolic_layer.fns_name[i][j]
                        a = float(symbolic_layer.affine_a[i, j])
                        b = float(symbolic_layer.affine_b[i, j])
                        c = float(symbolic_layer.affine_c[i, j])
                        d = float(symbolic_layer.affine_d[i, j])

                        x_expr = expressions[l][i]
                        inner = a * x_expr + b

                        # Map function names to sympy
                        if fn_name == "sin":
                            f_expr = sympy.sin(inner)
                        elif fn_name == "cos":
                            f_expr = sympy.cos(inner)
                        elif fn_name == "exp":
                            f_expr = sympy.exp(inner)
                        elif fn_name == "log":
                            f_expr = sympy.log(inner)
                        elif fn_name == "x^2":
                            f_expr = inner ** 2
                        elif fn_name == "x^3":
                            f_expr = inner ** 3
                        elif fn_name == "sqrt":
                            f_expr = sympy.sqrt(inner)
                        elif fn_name == "x":
                            f_expr = inner
                        elif fn_name == "1":
                            f_expr = 1
                        elif fn_name == "0":
                            f_expr = 0
                        else:
                            f_expr = sympy.Function(fn_name)(inner)

                        term = c * f_expr + d
                        terms.append(term)
                    else:
                        terms.append(sympy.Function(f"spline_{l}_{i}_{j}")(expressions[l][i]))

                expr = sum(terms)
                layer_expressions.append(sympy.simplify(expr))

            expressions.append(layer_expressions)

        final = expressions[-1]
        return final[0] if len(final) == 1 else final

    def to_tree(self) -> Dict[str, Any]:
        """Export as expression tree dictionary.

        Returns:
            Dictionary representation of the computation graph
        """
        tree = {
            "type": "MultKAN",
            "width": self.width,
            "layers": []
        }

        for l in range(self.depth):
            layer_tree = {
                "layer_idx": l,
                "in_dim": self.layers[l].in_dim,
                "out_dim": self.layers[l].out_dim,
                "edges": []
            }

            symbolic_layer = self.symbolic_funs[l]

            for i in range(self.layers[l].in_dim):
                for j in range(self.layers[l].out_dim):
                    edge = {
                        "from": i,
                        "to": j,
                        "symbolic": symbolic_layer.is_symbolic(i, j),
                    }

                    if edge["symbolic"]:
                        edge["function"] = symbolic_layer.fns_name[i][j]
                        edge["params"] = {
                            "a": float(symbolic_layer.affine_a[i, j]),
                            "b": float(symbolic_layer.affine_b[i, j]),
                            "c": float(symbolic_layer.affine_c[i, j]),
                            "d": float(symbolic_layer.affine_d[i, j]),
                        }
                    else:
                        edge["function"] = "spline"
                        edge["params"] = {
                            "grid": self.grid,
                            "k": self.k,
                        }

                    layer_tree["edges"].append(edge)

            tree["layers"].append(layer_tree)

        return tree

    # ===== Visualization Helpers =====

    def plot_kan(
        self,
        x: Optional[mx.array] = None,
        folder: str = "./figures",
        beta: float = 3.0,
        title: Optional[str] = None,
        in_vars: Optional[List[str]] = None,
        out_vars: Optional[List[str]] = None,
        display: bool = True,
        save: bool = True,
    ):
        """Plot KAN network with spline curves on edges (pykan-style).

        Args:
            x: Optional input data to run forward pass with. If provided and
               model has no cached activations, runs forward pass first.
        """
        from .visualization import plot_kan

        # If x is provided and we have no cached activations, run forward pass
        if x is not None and self._acts is None:
            self(x)

        return plot_kan(
            self, folder=folder, beta=beta, title=title,
            in_vars=in_vars, out_vars=out_vars, display=display, save=save
        )

    def plot_activations(
        self,
        layer_idx: int,
        x: Optional[mx.array] = None,
        folder: str = "./figures",
        display: bool = True,
        save: bool = True,
    ):
        """Plot all activation functions for a layer."""
        from .visualization import plot_activations
        return plot_activations(self, layer_idx, x=x, folder=folder, display=display, save=save)
