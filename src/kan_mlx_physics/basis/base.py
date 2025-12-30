"""Abstract base class for 1D basis functions in KAN layers.

This module provides the foundation for pluggable basis functions,
replacing the hardcoded B-spline with a modular architecture.

Design principles:
    1. Shape convention: phis (B, I, M), coef (I, O, M) -> output (B, O)
    2. Learnable affine parameters (shift, scale) per-input or per-edge
    3. Stable positive parameterization using softplus
    4. Smooth domain normalization using tanh/sigmoid instead of hard clip
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Literal
import mlx.core as mx


# Type alias for parameter mode
ParamMode = Literal["per_in", "per_edge"]


def softplus(x: mx.array, beta: float = 1.0, threshold: float = 20.0) -> mx.array:
    """Softplus activation: log(1 + exp(beta * x)) / beta.

    Numerically stable implementation that switches to linear for large x.

    Args:
        x: Input array
        beta: Steepness parameter (default 1.0)
        threshold: Switch to linear above this value for stability

    Returns:
        Softplus of x
    """
    scaled = beta * x
    # Use linear approximation for large values to avoid overflow
    return mx.where(
        scaled > threshold,
        x,
        mx.log1p(mx.exp(scaled)) / beta
    )


@dataclass
class BasisConfig:
    """Configuration for basis function initialization.

    Attributes:
        M: Number of basis functions
        learnable_affine: Whether shift/scale are learnable
        param_mode: "per_in" for shared params per input dim,
                   "per_edge" for separate params per (in, out) edge
        domain: Natural domain of the basis (for normalization)
        normalize_method: How to map inputs to domain ("tanh", "sigmoid", "none")
    """
    M: int
    learnable_affine: bool = True
    param_mode: ParamMode = "per_in"
    domain: tuple = (-1.0, 1.0)
    normalize_method: Literal["tanh", "sigmoid", "none"] = "tanh"


class Basis(ABC):
    """Abstract base class for 1D basis functions.

    Each basis provides M functions φ_m that map R -> R.
    The edge function becomes: f(x) = Σ_{m=0}^{M-1} c_m · φ_m((x - μ) / s)

    Shape conventions:
        - x: (batch, in_dim)
        - features output: (batch, in_dim, M)
        - coefficients: (in_dim, out_dim, M)
        - edge output: (batch, in_dim, out_dim)

    Attributes:
        name: Human-readable name for this basis type
        config: BasisConfig with initialization settings
    """

    name: str = "base"

    def __init__(self, config: BasisConfig):
        """Initialize basis with configuration.

        Args:
            config: BasisConfig specifying M, param_mode, etc.
        """
        self.config = config
        self.M = config.M

    @classmethod
    def from_M(cls, M: int, **kwargs) -> "Basis":
        """Convenience constructor from just M.

        Args:
            M: Number of basis functions
            **kwargs: Additional BasisConfig parameters

        Returns:
            Initialized Basis instance
        """
        config = BasisConfig(M=M, **kwargs)
        return cls(config)

    @property
    def num_features(self) -> int:
        """Number of basis functions."""
        return self.M

    def init_params(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
    ) -> Dict[str, mx.array]:
        """Initialize learnable parameters.

        Args:
            in_dim: Number of input dimensions
            out_dim: Number of output dimensions (needed for per_edge mode)

        Returns:
            Dictionary of parameters:
                - shift_raw: Raw shift values (apply softplus for positive)
                - scale_raw: Raw scale values (softplus -> scale)
        """
        params = {}

        if not self.config.learnable_affine:
            return params

        if self.config.param_mode == "per_in":
            # Shape: (in_dim,) - shared across all outputs
            params["shift"] = mx.zeros((in_dim,))
            params["scale_raw"] = mx.zeros((in_dim,))  # softplus(0) ≈ 0.693
        elif self.config.param_mode == "per_edge":
            if out_dim is None:
                raise ValueError("out_dim required for per_edge param_mode")
            # Shape: (in_dim, out_dim) - separate for each edge
            params["shift"] = mx.zeros((in_dim, out_dim))
            params["scale_raw"] = mx.zeros((in_dim, out_dim))

        # Subclasses can add additional params
        params.update(self._init_extra_params(in_dim, out_dim))

        return params

    def _init_extra_params(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
    ) -> Dict[str, mx.array]:
        """Initialize basis-specific extra parameters.

        Override in subclasses to add parameters like learnable frequencies.

        Args:
            in_dim: Number of input dimensions
            out_dim: Number of output dimensions

        Returns:
            Dictionary of extra parameters
        """
        return {}

    def _get_scale(self, params: Dict[str, Any]) -> Optional[mx.array]:
        """Get positive scale from raw params using softplus.

        Args:
            params: Parameter dictionary with scale_raw

        Returns:
            Positive scale values, or None if not learnable
        """
        scale_raw = params.get("scale_raw")
        if scale_raw is None:
            return None
        # softplus ensures positive scale, add small epsilon for stability
        return softplus(scale_raw) + 1e-6

    def _normalize(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Apply affine normalization: (x - shift) / scale.

        Args:
            x: Input of shape (batch, in_dim) or (batch, in_dim, out_dim)
            params: Parameter dictionary with shift and scale_raw

        Returns:
            Normalized input of same shape
        """
        shift = params.get("shift")
        scale = self._get_scale(params)

        if shift is not None:
            # Handle shape broadcasting for per_in vs per_edge
            if shift.ndim == 1 and x.ndim == 3:
                shift = shift[None, :, None]  # (1, I, 1)
            elif shift.ndim == 1:
                shift = shift[None, :]  # (1, I)
            elif shift.ndim == 2:
                shift = shift[None, :, :]  # (1, I, O)
            x = x - shift

        if scale is not None:
            if scale.ndim == 1 and x.ndim == 3:
                scale = scale[None, :, None]
            elif scale.ndim == 1:
                scale = scale[None, :]
            elif scale.ndim == 2:
                scale = scale[None, :, :]
            x = x / scale

        return x

    def _map_to_domain(self, x: mx.array) -> mx.array:
        """Map normalized x to the basis natural domain.

        Uses smooth mappings instead of hard clipping:
            - tanh: maps R -> (-1, 1), good for [-1, 1] domains
            - sigmoid: maps R -> (0, 1), good for [0, 1] domains
            - none: no mapping, basis handles its own domain

        Args:
            x: Normalized input

        Returns:
            Input mapped to basis domain
        """
        a, b = self.config.domain
        method = self.config.normalize_method

        if method == "none":
            return x
        elif method == "tanh":
            # tanh: R -> (-1, 1), then scale to (a, b)
            t = mx.tanh(x)
            return a + (b - a) * (t + 1) / 2
        elif method == "sigmoid":
            # sigmoid: R -> (0, 1), then scale to (a, b)
            s = mx.sigmoid(x)
            return a + (b - a) * s
        else:
            return x

    @abstractmethod
    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate basis functions at input points.

        Args:
            x: Input of shape (batch, in_dim)
            params: Parameter dictionary from init_params

        Returns:
            Basis values of shape (batch, in_dim, M)
        """
        pass

    def __call__(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Alias for features method."""
        return self.features(x, params)

    def symbolic(
        self,
        coeffs: mx.array,
        params: Dict[str, Any],
        var: str = "x",
    ) -> str:
        """Generate symbolic formula string.

        Args:
            coeffs: Coefficients of shape (M,) for one edge
            params: Parameter dictionary
            var: Variable name to use

        Returns:
            Human-readable formula string
        """
        return f"<{self.name} basis, M={self.M}>"

    @property
    def symbolic_priority(self) -> List[str]:
        """Functions to prioritize in symbolic regression.

        Returns a list of function names that this basis naturally represents,
        used to guide symbolic regression toward appropriate functions.
        """
        return []

    def orthonormalize(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Optionally orthonormalize basis functions (Gram-Schmidt).

        Default implementation returns features unchanged.
        Override for bases that benefit from orthonormalization.

        Args:
            x: Input of shape (batch, in_dim)
            params: Parameter dictionary

        Returns:
            Orthonormalized basis values of shape (batch, in_dim, M)
        """
        return self.features(x, params)


def contract_basis_coef(
    phis: mx.array,
    coef: mx.array,
) -> mx.array:
    """Contract basis features with coefficients.

    Implements: output[b, i, o] = Σ_m phis[b, i, m] * coef[i, o, m]

    Args:
        phis: Basis features of shape (batch, in_dim, M)
        coef: Coefficients of shape (in_dim, out_dim, M)

    Returns:
        Edge outputs of shape (batch, in_dim, out_dim)
    """
    # phis: (B, I, M) -> (B, I, 1, M)
    # coef: (I, O, M) -> (1, I, O, M)
    phis_exp = mx.expand_dims(phis, axis=2)
    coef_exp = mx.expand_dims(coef, axis=0)

    # Element-wise multiply and sum over M
    return mx.sum(phis_exp * coef_exp, axis=-1)  # (B, I, O)
