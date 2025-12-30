"""KAN Layer implementation for MLX with pluggable basis functions.

Supports multiple basis types:
    - B-splines (default, backwards compatible)
    - Fourier (periodic/oscillatory)
    - Chebyshev (spectral methods)
    - Hermite (quantum mechanics)
    - Laguerre (radial problems)
    - Legendre (angular momentum)

Example:
    # Traditional B-spline layer
    layer = KANLayer(in_dim=5, out_dim=3)

    # FourierKAN layer
    layer = KANLayer(in_dim=5, out_dim=3, basis="fourier", basis_M=11)

    # HermiteKAN for quantum mechanics
    layer = KANLayer(in_dim=5, out_dim=3, basis="hermite", basis_kwargs={"weighted": True})
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, Optional, Callable, List, overload, Literal, Union, Dict, Any

from .spline import B_batch, coef2curve, curve2coef, extend_grid


class ActivationCache(Tuple):
    """Cache of intermediate activations from a forward pass.

    Attributes:
        preacts: Pre-activations of shape (batch, out_dim, in_dim)
        postacts: Post-activations of shape (batch, in_dim, out_dim)
        postspline: Spline outputs of shape (batch, in_dim, out_dim)
    """
    pass


class KANLayer(nn.Module):
    """Single Kolmogorov-Arnold Network layer with pluggable basis functions.

    Each edge (i, j) has a learnable univariate function parameterized by
    a configurable basis (B-splines, Fourier, Chebyshev, Hermite, etc.).

    The output is computed as: y_j = sum_i phi_{i,j}(x_i)

    where phi_{i,j}(x) = scale_sp * sum_m c_{ijm} * basis_m(x) + scale_base * base(x)

    Args:
        in_dim: Number of input neurons
        out_dim: Number of output neurons
        num_grid: Number of grid intervals for B-splines (ignored if basis != "bspline")
        k: Spline polynomial order (default: 3 for cubic, ignored if basis != "bspline")
        noise_scale: Scale of noise for coefficient initialization
        base_fun: Base activation function for residual connection (default: SiLU
            provides smooth gradients and avoids dead neurons unlike ReLU)
        grid_range: Range for uniform grid initialization (B-spline only)
        basis: Basis type name. Options:
            - "bspline": B-splines (default, general purpose)
            - "fourier": Fourier basis (periodic/oscillatory, wave equations)
            - "chebyshev": Chebyshev polynomials (spectral methods, bounded intervals)
            - "hermite": Hermite polynomials (quantum harmonic oscillator)
            - "laguerre": Laguerre polynomials (radial problems, hydrogen atom)
            - "legendre": Legendre polynomials (angular momentum, spherical harmonics)
        basis_M: Number of basis functions. If None, uses num_grid + k.
            For Fourier: M=11 gives 5 harmonics (1 + 5 cos + 5 sin terms).
        basis_kwargs: Basis-specific parameters. Options by basis type:
            - fourier: {"learnable_freq": bool (default True), "base_freq": float (default 1.0)}
            - hermite: {"weighted": bool (default False) - multiply by exp(-x²/2) for QM}
            - laguerre: {"alpha": float (default 0.0) - generalized Laguerre L_n^α}
            - chebyshev/legendre: {"learnable_affine": bool (default True)}

    Example:
        # Standard B-spline layer
        layer = KANLayer(in_dim=5, out_dim=3)

        # Fourier for periodic functions
        layer = KANLayer(in_dim=5, out_dim=3, basis="fourier", basis_M=11)

        # Hermite for quantum mechanics (with Gaussian weight)
        layer = KANLayer(in_dim=5, out_dim=3, basis="hermite",
                        basis_kwargs={"weighted": True})
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_grid: int = 5,
        k: int = 3,
        noise_scale: float = 0.1,
        base_fun: Callable = nn.silu,
        grid_range: Tuple[float, float] = (-1.0, 1.0),
        # New basis parameters
        basis: str = "bspline",
        basis_M: Optional[int] = None,
        basis_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_grid = num_grid
        self.k = k
        self.base_fun = base_fun
        self.basis_type = basis

        # For backwards compatibility, store grid_range
        self._grid_range = grid_range

        # Initialize basis
        basis_kwargs = basis_kwargs or {}

        # Validate basis name early to give clear error messages
        valid_bases = {"bspline", "fourier", "chebyshev", "hermite", "laguerre", "legendre"}
        if basis not in valid_bases:
            raise ValueError(
                f"Unknown basis '{basis}'. Valid options: {sorted(valid_bases)}"
            )

        if basis == "bspline":
            # Backwards compatible B-spline initialization
            # IMPORTANT: For B-splines, always use num_grid + k (ignore basis_M)
            # This ensures grid and coefficient dimensions match
            num_coef = num_grid + k

            # Initialize uniform grid and extend for spline boundaries
            grid = mx.linspace(grid_range[0], grid_range[1], num_grid + 1)
            grid = mx.broadcast_to(grid, (in_dim, num_grid + 1))
            grid = extend_grid(grid, k_extend=k)
            self._grid = grid  # (in_dim, num_grid + 2*k + 1)

            self._basis = None  # Use legacy code path for B-splines

            # No trainable basis params for B-splines
            self.basis_shift = None
            self.basis_scale_raw = None
            self.basis_omega_raw = None
        else:
            # Use new pluggable basis system
            from .basis import make_basis

            M = basis_M if basis_M is not None else (num_grid + k)
            # Force per-input params only (not per-edge) for simplicity
            basis_obj = make_basis(basis, M=M, param_mode="per_in", **basis_kwargs)
            raw_params = basis_obj.init_params(in_dim)  # per-input only
            num_coef = basis_obj.num_features

            # Store basis features function (not the whole object to avoid MLX traversal issues)
            # We store it in a private tuple that MLX won't traverse
            self.__basis_info = (basis_obj, basis_obj.features, basis_obj.symbolic_priority)

            # Store basis params as direct attributes so MLX picks them up
            # This makes them trainable via model.parameters()
            self.basis_shift = raw_params.get("shift")
            self.basis_scale_raw = raw_params.get("scale_raw")
            self.basis_omega_raw = raw_params.get("omega_raw")

            # Placeholder for grid (not used for non-B-spline bases)
            self._grid = None

        # Store number of basis features
        self._num_basis_features = num_coef

        # Learnable parameters
        # Coefficients with small random initialization
        self.coef = mx.random.uniform(
            low=-noise_scale,
            high=noise_scale,
            shape=(in_dim, out_dim, num_coef),
        )

        # Scale factors for spline and base functions
        self.scale_sp = mx.ones((in_dim, out_dim))
        self.scale_base = mx.ones((in_dim, out_dim))

        # Mask for pruning (1 = active, 0 = pruned)
        self._mask = mx.ones((in_dim, out_dim))

    @property
    def grid(self) -> mx.array:
        """Get the current grid (non-trainable buffer)."""
        return self._grid

    @property
    def mask(self) -> mx.array:
        """Get the current mask (non-trainable buffer)."""
        return self._mask

    @overload
    def __call__(
        self,
        x: mx.array,
        return_activations: Literal[False] = False,
    ) -> mx.array: ...

    @overload
    def __call__(
        self,
        x: mx.array,
        return_activations: Literal[True],
    ) -> Tuple[mx.array, mx.array, mx.array, mx.array]: ...

    def __call__(
        self,
        x: mx.array,
        return_activations: bool = False,
    ) -> Union[mx.array, Tuple[mx.array, mx.array, mx.array, mx.array]]:
        """Forward pass through the KAN layer.

        Args:
            x: Input tensor of shape (batch, in_dim)
            return_activations: If True, return intermediate activations

        Returns:
            If return_activations is False:
                Output tensor of shape (batch, out_dim)
            If return_activations is True:
                Tuple of (output, preacts, postacts, postspline)
        """
        batch_size = x.shape[0]

        # Pre-activations: just the input repeated for each output
        # Shape: (batch, out_dim, in_dim) for visualization purposes
        preacts = mx.broadcast_to(
            mx.expand_dims(x, axis=1),
            (batch_size, self.out_dim, self.in_dim),
        )

        # Base activation (residual connection)
        # Shape: (batch, in_dim)
        base = self.base_fun(x)

        # Evaluate basis functions and compute edge outputs
        # The goal is to compute: spline[b, i, j] = Σ_m coef[i, j, m] * φ_m(x[b, i])
        # where φ_m are the basis functions (B-spline, Fourier, etc.)
        if not hasattr(self, '_KANLayer__basis_info'):
            # Legacy B-spline code path for backwards compatibility
            # coef2curve handles the basis evaluation and contraction internally
            # Output shape: (batch, in_dim, out_dim)
            spline = coef2curve(x, self._grid, self.coef, self.k)
        else:
            # New pluggable basis system
            # Build params dict from class attributes (stored separately for MLX compatibility)
            basis_params = {}
            if self.basis_shift is not None:
                basis_params["shift"] = self.basis_shift      # Per-input shift: (in_dim,)
            if self.basis_scale_raw is not None:
                basis_params["scale_raw"] = self.basis_scale_raw  # Per-input scale: (in_dim,)
            if self.basis_omega_raw is not None:
                basis_params["omega_raw"] = self.basis_omega_raw  # Fourier frequency: (in_dim,)

            # Evaluate basis functions at each input
            # phis[b, i, m] = φ_m(x[b, i]) where m indexes basis functions
            basis_obj, basis_features_fn, _ = self.__basis_info
            phis = basis_features_fn(x, basis_params)  # Shape: (batch, in_dim, M)

            # Contract basis features with coefficients via einsum-like operation:
            # spline[b, i, j] = Σ_m phis[b, i, m] * coef[i, j, m]
            # We broadcast: phis (B, I, 1, M) * coef (1, I, O, M) -> sum over M
            phis_exp = mx.expand_dims(phis, axis=2)        # (B, I, 1, M)
            coef_exp = mx.expand_dims(self.coef, axis=0)   # (1, I, O, M)
            spline = mx.sum(phis_exp * coef_exp, axis=-1)  # (B, I, O)

        # Combine base activation and spline with learned scales
        # Each edge (i, j) has its own scale factors: scale_base[i,j] and scale_sp[i,j]
        # Final edge function: f_{ij}(x_i) = scale_base * base(x_i) + scale_sp * spline(x_i)
        scale_base = mx.expand_dims(self.scale_base, axis=0)  # (1, I, O) for broadcasting
        scale_sp = mx.expand_dims(self.scale_sp, axis=0)      # (1, I, O) for broadcasting
        base_expanded = mx.expand_dims(base, axis=2)          # (B, I, 1) for broadcasting

        # Compute per-edge outputs: y[b, i, j] = scale_base[i,j] * base[b,i] + scale_sp[i,j] * spline[b,i,j]
        y = scale_base * base_expanded + scale_sp * spline    # Shape: (B, I, O)

        # Apply pruning mask (0 = pruned edge, 1 = active edge)
        mask = mx.expand_dims(self._mask, axis=0)  # (1, I, O)
        y = y * mask

        # Store activations for visualization/analysis
        postspline = spline  # Raw spline output before scaling
        postacts = y         # Full edge outputs after scaling and masking

        # KAN layer output: sum over all input edges
        # output[b, j] = Σ_i y[b, i, j] (Kolmogorov-Arnold superposition)
        output = mx.sum(y, axis=1)  # Shape: (B, O)

        if return_activations:
            return output, preacts, postacts, postspline

        return output

    def update_grid_from_samples(
        self,
        x: mx.array,
        margin: float = 0.01,
        grid_eps: float = 0.02,
    ) -> None:
        """Update the grid based on input data distribution.

        Uses percentile-based grid placement mixed with uniform grid.
        Only applies to B-spline basis; other bases are skipped.

        Args:
            x: Input samples of shape (batch, in_dim)
            margin: Margin to extend grid beyond data range
            grid_eps: Interpolation between uniform (0) and adaptive (1) grid
        """
        # Skip for non-B-spline bases (they don't use adaptive grids)
        if self._grid is None:
            return

        batch_size = x.shape[0]

        # Compute data range per input dimension
        x_min = mx.min(x, axis=0)  # (in_dim,)
        x_max = mx.max(x, axis=0)  # (in_dim,)

        # Add margin
        x_range = x_max - x_min
        x_min = x_min - margin * x_range
        x_max = x_max + margin * x_range

        # Create uniform grid
        uniform_grid = []
        for i in range(self.in_dim):
            uniform_grid.append(
                mx.linspace(x_min[i].item(), x_max[i].item(), self.num_grid + 1)
            )
        uniform_grid = mx.stack(uniform_grid, axis=0)  # (in_dim, num_grid + 1)

        # For adaptive grid, we would compute percentiles
        # For now, use uniform grid (percentile computation is complex in MLX)
        # TODO: Add percentile-based adaptive grid

        new_grid = uniform_grid

        # Extend grid for spline boundaries
        new_grid = extend_grid(new_grid, k_extend=self.k)

        # Update grid and refit coefficients
        self._update_grid_and_coef(x, new_grid)

    def _update_grid_and_coef(self, x: mx.array, new_grid: mx.array) -> None:
        """Update grid and refit coefficients to preserve function values.

        Args:
            x: Input samples for refitting
            new_grid: New grid to use
        """
        # Evaluate current spline at sample points
        y = coef2curve(x, self._grid, self.coef, self.k)  # (batch, in_dim, out_dim)

        # Update grid
        self._grid = new_grid

        # Refit coefficients to match previous function values
        self.coef = curve2coef(x, y, new_grid, self.k)

    def set_mask(self, mask: mx.array) -> None:
        """Set the pruning mask.

        Args:
            mask: Binary mask of shape (in_dim, out_dim)
        """
        self._mask = mask

    def get_subset(
        self,
        in_ids: List[int],
        out_ids: List[int],
    ) -> "KANLayer":
        """Extract a subset of the layer for pruning.

        Args:
            in_ids: Indices of input neurons to keep
            out_ids: Indices of output neurons to keep

        Returns:
            New KANLayer with the specified subset
        """
        # For B-splines, use num_grid; for others, use basis_M
        basis_M_arg = None if self.basis_type == "bspline" else self._num_basis_features

        new_layer = KANLayer(
            in_dim=len(in_ids),
            out_dim=len(out_ids),
            num_grid=self.num_grid,
            k=self.k,
            base_fun=self.base_fun,
            basis=self.basis_type,
            basis_M=basis_M_arg,
        )

        # Copy relevant parameters
        in_ids_arr = mx.array(in_ids)
        out_ids_arr = mx.array(out_ids)

        # Slice coefficients
        new_coef = self.coef[in_ids_arr, :, :][:, out_ids_arr, :]
        new_layer.coef = new_coef

        # Slice scales
        new_layer.scale_sp = self.scale_sp[in_ids_arr, :][:, out_ids_arr]
        new_layer.scale_base = self.scale_base[in_ids_arr, :][:, out_ids_arr]

        # Slice grid (B-spline only)
        if self._grid is not None:
            new_layer._grid = self._grid[in_ids_arr, :]

        # Slice mask
        new_layer._mask = self._mask[in_ids_arr, :][:, out_ids_arr]

        # Handle basis params (per-input only, shape (in_dim,))
        if self.basis_shift is not None:
            new_layer.basis_shift = self.basis_shift[in_ids_arr]
        if self.basis_scale_raw is not None:
            new_layer.basis_scale_raw = self.basis_scale_raw[in_ids_arr]
        if self.basis_omega_raw is not None:
            new_layer.basis_omega_raw = self.basis_omega_raw[in_ids_arr]

        return new_layer

    def edge_scores(self) -> mx.array:
        """Compute importance scores for each edge.

        Returns:
            Scores of shape (in_dim, out_dim)
        """
        # Simple scoring based on coefficient magnitude and scale
        coef_norm = mx.mean(mx.abs(self.coef), axis=2)  # (in_dim, out_dim)
        scores = coef_norm * mx.abs(self.scale_sp) + mx.abs(self.scale_base)
        return scores * self._mask

    @property
    def symbolic_priority(self) -> List[str]:
        """Get symbolic function priority from basis type.

        Returns:
            List of function names to prioritize in symbolic regression
        """
        if hasattr(self, '_KANLayer__basis_info'):
            _, _, priority = self.__basis_info
            return priority
        # Default for B-splines
        return ["x", "x^2", "x^3", "x^4", "sin", "cos", "exp", "log", "tanh"]

    @property
    def basis_params(self) -> Dict[str, mx.array]:
        """Get the basis parameters (for training/inspection).

        Returns:
            Dictionary of basis parameters (shift, scale, omega, etc.)
        """
        params = {}
        if self.basis_shift is not None:
            params["shift"] = self.basis_shift
        if self.basis_scale_raw is not None:
            params["scale_raw"] = self.basis_scale_raw
        if self.basis_omega_raw is not None:
            params["omega_raw"] = self.basis_omega_raw
        return params

    @property
    def num_basis_features(self) -> int:
        """Number of basis functions per edge."""
        return self._num_basis_features
