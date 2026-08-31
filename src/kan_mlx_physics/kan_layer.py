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
        # Mixed-basis mult slot parameters
        basis_per_mult_slot: Optional[List[str]] = None,
        n_sum_out: Optional[int] = None,
        n_mult_out: Optional[int] = None,
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

        # Validate basis name early to give clear error messages. Validate
        # against the actual basis registry (which includes physics variants
        # like weighted_hermite, associated_laguerre, hydrogen_radial, …) rather
        # than a hardcoded subset — otherwise usable bases get spuriously
        # rejected. "bspline" is handled by the legacy path below.
        from .basis import list_bases
        valid_bases = set(list_bases()) | {"bspline"}
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

            # Override affine init for range-sensitive polynomial bases so inputs
            # in grid_range are mapped to [-1, 1] instead of being passed raw.
            _range_sensitive = {"hermite", "chebyshev", "legendre"}
            if basis in _range_sensitive:
                _lo, _hi = grid_range
                _center = (_lo + _hi) / 2.0
                _half = max((_hi - _lo) / 2.0, 1e-3)
                import math as _math
                _sr_init = _math.log(_math.exp(_half - 1e-6) - 1.0)
                if raw_params.get("shift") is not None:
                    raw_params["shift"] = mx.full((in_dim,), _center)
                if raw_params.get("scale_raw") is not None:
                    raw_params["scale_raw"] = mx.full((in_dim,), _sr_init)

            # Store basis params as direct attributes so MLX picks them up
            # This makes them trainable via model.parameters()
            self.basis_shift = raw_params.get("shift")
            self.basis_scale_raw = raw_params.get("scale_raw")
            self.basis_omega_raw = raw_params.get("omega_raw")
            self.basis_centers = raw_params.get("centers")
            self.basis_log_widths = raw_params.get("log_widths")

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

        # Mixed-basis mult slot support
        # When basis_per_mult_slot is set, mult-node input edges (indices n_sum_out onwards)
        # are split into groups — one per slot — each evaluated with a different basis.
        self._basis_per_mult_slot = basis_per_mult_slot
        self._n_sum_out = n_sum_out
        self._n_mult_out = n_mult_out

        if basis_per_mult_slot is not None:
            if n_sum_out is None or n_mult_out is None:
                raise ValueError(
                    "n_sum_out and n_mult_out must be provided when basis_per_mult_slot is set"
                )
            mult_arity = len(basis_per_mult_slot)

            from .basis import make_basis

            # Validate slot basis names
            for slot_basis in basis_per_mult_slot:
                if slot_basis not in valid_bases:
                    raise ValueError(
                        f"Unknown slot basis '{slot_basis}'. Valid options: {sorted(valid_bases)}"
                    )

            M_slot = basis_M if basis_M is not None else (num_grid + k)

            # Build per-slot basis objects; store in a private tuple (not traversed by MLX)
            slot_basis_objs = []
            for slot_basis in basis_per_mult_slot:
                if slot_basis == "bspline":
                    # B-spline slot — use same num_coef as main basis for simplicity
                    # We create a dummy basis object wrapper for the forward pass
                    slot_obj = None  # special-cased below
                else:
                    slot_obj = make_basis(slot_basis, M=M_slot, param_mode="per_in")
                slot_basis_objs.append(slot_obj)

            self.__slot_basis_info = tuple(
                (obj, obj.features, obj.symbolic_priority) if obj is not None else None
                for obj in slot_basis_objs
            )
            self._slot_basis_types = basis_per_mult_slot  # store for inspection

            # Polynomial bases (hermite, chebyshev, legendre) use learnable affine
            # params (shift, scale_raw) to map inputs into their natural domain.
            # With scale_raw=0 (default), softplus(0)≈0.693 — too small for [0,6]
            # inputs, causing polynomial explosion and NaN from step 0.
            # We initialize shift/scale_raw to map grid_range to [-1, 1]:
            #   shift  = (lo + hi) / 2     (center of the domain)
            #   scale  = (hi - lo) / 2     (half-width)
            #   scale_raw = softplus_inv(scale - 1e-6) ≈ scale for scale >> 0
            # This is only applied to bases whose stability depends on input range.
            _range_sensitive_bases = {"hermite", "chebyshev", "legendre"}
            _gr_lo, _gr_hi = grid_range
            _gr_center = (_gr_lo + _gr_hi) / 2.0
            _gr_half = max((_gr_hi - _gr_lo) / 2.0, 1e-3)
            # softplus_inv(y) ≈ log(exp(y) - 1); for y > 1 this ≈ y
            import math as _math
            _scale_raw_init = _math.log(_math.exp(_gr_half - 1e-6) - 1.0)

            # Per-slot per-input basis params stored as flat lists of attributes.
            # Naming convention: slot_basis_shift_0, slot_basis_shift_1, ...
            # so that MLX can traverse and train them.
            for s, slot_obj in enumerate(slot_basis_objs):
                if slot_obj is None:
                    # B-spline slot: initialize a grid per slot
                    slot_grid = mx.linspace(grid_range[0], grid_range[1], num_grid + 1)
                    slot_grid = mx.broadcast_to(slot_grid, (in_dim, num_grid + 1))
                    slot_grid = extend_grid(slot_grid, k_extend=k)
                    setattr(self, f"_slot_grid_{s}", slot_grid)
                else:
                    slot_raw = slot_obj.init_params(in_dim)

                    # Override affine init for range-sensitive polynomial bases
                    if basis_per_mult_slot[s] in _range_sensitive_bases:
                        if slot_raw.get("shift") is not None:
                            slot_raw["shift"] = mx.full((in_dim,), _gr_center)
                        if slot_raw.get("scale_raw") is not None:
                            slot_raw["scale_raw"] = mx.full((in_dim,), _scale_raw_init)

                    setattr(self, f"slot_basis_shift_{s}", slot_raw.get("shift"))
                    setattr(self, f"slot_basis_scale_raw_{s}", slot_raw.get("scale_raw"))
                    setattr(self, f"slot_basis_omega_raw_{s}", slot_raw.get("omega_raw"))
                    setattr(self, f"slot_basis_centers_{s}", slot_raw.get("centers"))
                    setattr(self, f"slot_basis_log_widths_{s}", slot_raw.get("log_widths"))

            # Per-slot coefficient tensors.
            # Each slot has n_mult_out output edges and uses its own basis features.
            # coef shape per slot: (in_dim, n_mult_out, num_slot_coef)
            for s, slot_obj in enumerate(slot_basis_objs):
                if slot_obj is None:
                    n_slot_coef = num_grid + k
                else:
                    n_slot_coef = slot_obj.num_features
                slot_coef = mx.random.uniform(
                    low=-noise_scale,
                    high=noise_scale,
                    shape=(in_dim, n_mult_out, n_slot_coef),
                )
                setattr(self, f"slot_coef_{s}", slot_coef)

            # The main self.coef covers only sum-node outputs now.
            # We replace it with a sum-only coef of shape (in_dim, n_sum_out, num_coef).
            # (If n_sum_out == 0, the main coef is unused but kept for shape consistency.)
            self.coef = mx.random.uniform(
                low=-noise_scale,
                high=noise_scale,
                shape=(in_dim, n_sum_out, num_coef),
            )
            # scale_sp and scale_base also shrink to sum-only outputs
            self.scale_sp = mx.ones((in_dim, n_sum_out))
            self.scale_base = mx.ones((in_dim, n_sum_out))
            # Per-slot scale params
            for s in range(mult_arity):
                setattr(self, f"slot_scale_sp_{s}", mx.ones((in_dim, n_mult_out)))
                setattr(self, f"slot_scale_base_{s}", mx.ones((in_dim, n_mult_out)))

    @property
    def grid(self) -> mx.array:
        """Get the current grid (non-trainable buffer)."""
        return self._grid

    @property
    def mask(self) -> mx.array:
        """Get the current mask (non-trainable buffer)."""
        return self._mask

    @property
    def _slot_bases(self) -> Optional[List[str]]:
        """Slot basis type names, or None if not using mixed-basis mult slots."""
        return self._basis_per_mult_slot

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
        base_expanded = mx.expand_dims(base, axis=2)  # (B, I, 1)

        # ---------------------------------------------------------------
        # Mixed-basis mult-slot path
        # ---------------------------------------------------------------
        if self._basis_per_mult_slot is not None:
            n_sum_out = self._n_sum_out
            n_mult_out = self._n_mult_out
            mult_arity = len(self._basis_per_mult_slot)

            partial_outputs = []  # list of (B, I, n_out) tensors to concat along axis=2

            # --- Sum-node edges (main basis) ---
            if n_sum_out > 0:
                if not hasattr(self, '_KANLayer__basis_info'):
                    spline_sum = coef2curve(x, self._grid, self.coef, self.k)
                else:
                    basis_params = {}
                    if self.basis_shift is not None:
                        basis_params["shift"] = self.basis_shift
                    if self.basis_scale_raw is not None:
                        basis_params["scale_raw"] = self.basis_scale_raw
                    if self.basis_omega_raw is not None:
                        basis_params["omega_raw"] = self.basis_omega_raw
                    if self.basis_centers is not None:
                        basis_params["centers"] = self.basis_centers
                    if self.basis_log_widths is not None:
                        basis_params["log_widths"] = self.basis_log_widths
                    _, basis_features_fn, _ = self.__basis_info
                    phis = basis_features_fn(x, basis_params)
                    phis_exp = mx.expand_dims(phis, axis=2)
                    coef_exp = mx.expand_dims(self.coef, axis=0)
                    spline_sum = mx.sum(phis_exp * coef_exp, axis=-1)  # (B, I, n_sum_out)

                scale_base_s = mx.expand_dims(self.scale_base, axis=0)
                scale_sp_s = mx.expand_dims(self.scale_sp, axis=0)
                y_sum = scale_base_s * base_expanded + scale_sp_s * spline_sum
                partial_outputs.append(y_sum)

            # --- Per-slot mult-node edges ---
            for s in range(mult_arity):
                slot_basis_type = self._basis_per_mult_slot[s]
                slot_coef = getattr(self, f"slot_coef_{s}")  # (I, n_mult_out, M_s)
                slot_scale_sp = getattr(self, f"slot_scale_sp_{s}")    # (I, n_mult_out)
                slot_scale_base = getattr(self, f"slot_scale_base_{s}")

                if slot_basis_type == "bspline":
                    slot_grid = getattr(self, f"_slot_grid_{s}")
                    spline_slot = coef2curve(x, slot_grid, slot_coef, self.k)
                else:
                    slot_info_entry = self.__slot_basis_info[s]
                    _, slot_features_fn, _ = slot_info_entry

                    slot_params = {}
                    v = getattr(self, f"slot_basis_shift_{s}", None)
                    if v is not None:
                        slot_params["shift"] = v
                    v = getattr(self, f"slot_basis_scale_raw_{s}", None)
                    if v is not None:
                        slot_params["scale_raw"] = v
                    v = getattr(self, f"slot_basis_omega_raw_{s}", None)
                    if v is not None:
                        slot_params["omega_raw"] = v
                    v = getattr(self, f"slot_basis_centers_{s}", None)
                    if v is not None:
                        slot_params["centers"] = v
                    v = getattr(self, f"slot_basis_log_widths_{s}", None)
                    if v is not None:
                        slot_params["log_widths"] = v

                    phis_s = slot_features_fn(x, slot_params)        # (B, I, M_s)
                    phis_s_exp = mx.expand_dims(phis_s, axis=2)      # (B, I, 1, M_s)
                    coef_s_exp = mx.expand_dims(slot_coef, axis=0)   # (1, I, n_mult_out, M_s)
                    spline_slot = mx.sum(phis_s_exp * coef_s_exp, axis=-1)  # (B, I, n_mult_out)

                slot_scale_sp_e = mx.expand_dims(slot_scale_sp, axis=0)
                slot_scale_base_e = mx.expand_dims(slot_scale_base, axis=0)
                y_slot = slot_scale_base_e * base_expanded + slot_scale_sp_e * spline_slot
                partial_outputs.append(y_slot)

            # Concat along output dimension: (B, I, n_sum_out + mult_arity * n_mult_out)
            y = mx.concatenate(partial_outputs, axis=2)

            # Apply pruning mask
            mask = mx.expand_dims(self._mask, axis=0)
            y = y * mask

            postspline = y
            postacts = y
            output = mx.sum(y, axis=1)

            if return_activations:
                return output, preacts, postacts, postspline
            return output

        # ---------------------------------------------------------------
        # Standard path (no mixed-basis slots)
        # ---------------------------------------------------------------

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
            if self.basis_centers is not None:
                basis_params["centers"] = self.basis_centers       # Gaussian centers: (in_dim, M)
            if self.basis_log_widths is not None:
                basis_params["log_widths"] = self.basis_log_widths  # Gaussian log-widths: (in_dim, M)

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
        adaptive_weight: float = 0.5,
        quantiles: tuple = (0.01, 0.99),
    ) -> None:
        """Update the grid based on input data distribution.

        Uses hybrid uniform + percentile-based grid placement for adaptive
        knot distribution. This is the KAN paper's "grid update" trick.

        Args:
            x: Input samples of shape (batch, in_dim)
            margin: Margin to extend grid beyond data range
            adaptive_weight: Weight for adaptive (percentile) vs uniform grid.
                0.0 = pure uniform, 1.0 = pure adaptive, 0.5 = balanced blend
            quantiles: (low, high) quantiles for clipping extreme values
        """
        import numpy as np

        # Skip for non-B-spline bases (they don't use adaptive grids)
        if self._grid is None:
            return

        # Convert to numpy for percentile computation
        x_np = np.array(x)

        # Compute quantile-based range (more robust than min/max)
        q_low = np.percentile(x_np, quantiles[0] * 100, axis=0)
        q_high = np.percentile(x_np, quantiles[1] * 100, axis=0)

        # Add margin
        x_range = q_high - q_low
        x_min = q_low - margin * x_range
        x_max = q_high + margin * x_range

        # Build hybrid grid per input dimension
        new_grid_list = []
        for i in range(self.in_dim):
            # Percentile positions for knots (adaptive grid)
            percentiles = np.linspace(0, 100, self.num_grid + 1)
            adaptive_knots = np.percentile(x_np[:, i], percentiles)

            # Clip adaptive knots to the quantile range
            adaptive_knots = np.clip(adaptive_knots, x_min[i], x_max[i])

            # Uniform knots within the quantile range
            uniform_knots = np.linspace(x_min[i], x_max[i], self.num_grid + 1)

            # Blend adaptive and uniform grids
            blended_knots = (
                adaptive_weight * adaptive_knots +
                (1 - adaptive_weight) * uniform_knots
            )

            # Ensure monotonicity (knots must be increasing)
            blended_knots = np.sort(blended_knots)

            new_grid_list.append(mx.array(blended_knots))

        new_grid = mx.stack(new_grid_list, axis=0)  # (in_dim, num_grid + 1)

        # Extend grid for spline boundaries
        new_grid = extend_grid(new_grid, k_extend=self.k)

        # Update grid and refit coefficients
        self._update_grid_and_coef(x, new_grid)

    def extend_grid_resolution(self, factor: int = 2) -> None:
        """Increase grid resolution by factor, interpolating coefficients.

        This is the KAN paper's "grid extension" trick for curriculum training:
        start with coarse grid, then refine progressively.

        Args:
            factor: Multiplication factor for grid points (e.g., 2 doubles resolution)
        """
        import numpy as np

        # Skip for non-B-spline bases
        if self._grid is None:
            return

        old_num_grid = self.num_grid
        new_num_grid = old_num_grid * factor

        # Get old grid without boundary extensions
        old_grid = self._grid  # (in_dim, old_num_grid + 2*k + 1)

        # Interior knots only (exclude extended boundary)
        interior_start = self.k
        interior_end = old_num_grid + self.k + 1
        old_interior = np.array(old_grid[:, interior_start:interior_end])  # (in_dim, old_num_grid + 1)

        # Interpolate to finer grid
        new_interior = []
        for i in range(self.in_dim):
            old_knots = old_interior[i]
            # Linear interpolation to new resolution
            new_positions = np.linspace(0, len(old_knots) - 1, new_num_grid + 1)
            new_knots = np.interp(
                new_positions,
                np.arange(len(old_knots)),
                old_knots
            )
            new_interior.append(mx.array(new_knots))

        new_grid = mx.stack(new_interior, axis=0)
        new_grid = extend_grid(new_grid, k_extend=self.k)

        # Interpolate coefficients
        # Old coef: (in_dim, out_dim, old_num_grid + k)
        # New coef: (in_dim, out_dim, new_num_grid + k)
        old_coef = np.array(self.coef)
        old_num_coef = old_num_grid + self.k
        new_num_coef = new_num_grid + self.k

        new_coef = np.zeros((self.in_dim, self.out_dim, new_num_coef))
        for i in range(self.in_dim):
            for j in range(self.out_dim):
                # Linear interpolation of coefficients
                new_positions = np.linspace(0, old_num_coef - 1, new_num_coef)
                new_coef[i, j] = np.interp(
                    new_positions,
                    np.arange(old_num_coef),
                    old_coef[i, j]
                )

        # Update layer state
        self.num_grid = new_num_grid
        self._grid = new_grid
        self.coef = mx.array(new_coef)

    def _update_grid_and_coef(self, x: mx.array, new_grid: mx.array) -> None:
        """Update grid and refit coefficients to preserve function values.

        Args:
            x: Input samples for refitting
            new_grid: New grid to use
        """
        # Evaluate current spline at sample points
        y = coef2curve(x, self._grid, self.coef, self.k)  # (batch, in_dim, out_dim)
        mx.eval(y)  # CRITICAL: Sync before grid change to prevent Metal recompilation issues

        # Update grid
        self._grid = new_grid

        # Refit coefficients to match previous function values
        new_coef = curve2coef(x, y, new_grid, self.k)
        mx.eval(new_coef)  # CRITICAL: Sync after refit

        self.coef = new_coef

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
            Scores of shape (in_dim, total_out_dim) where total_out_dim
            includes both sum edges and slot edges.
        """
        parts = []

        # Sum-node edges (main basis)
        if self.out_dim > 0:
            coef_norm = mx.mean(mx.abs(self.coef), axis=2)  # (in_dim, out_dim)
            scores = coef_norm * mx.abs(self.scale_sp) + mx.abs(self.scale_base)
            # Apply mask only where mask shape matches
            if self._mask.shape == scores.shape:
                scores = scores * self._mask
            parts.append(scores)

        # Slot edges (one per slot per mult node)
        if self._slot_bases is not None:
            for s, _ in enumerate(self._slot_bases):
                slot_coef = getattr(self, f"slot_coef_{s}", None)
                slot_scale_sp = getattr(self, f"slot_scale_sp_{s}", None)
                slot_scale_base = getattr(self, f"slot_scale_base_{s}", None)
                if slot_coef is not None:
                    slot_norm = mx.mean(mx.abs(slot_coef), axis=2)
                    parts.append(slot_norm * mx.abs(slot_scale_sp) + mx.abs(slot_scale_base))

        if not parts:
            return mx.zeros((self.in_dim, 0))
        return mx.concatenate(parts, axis=1)

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
