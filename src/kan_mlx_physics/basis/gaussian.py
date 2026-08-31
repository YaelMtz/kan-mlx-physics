"""Gaussian radial basis functions (RBFs) for KAN edges.

Natural basis for:
    - QHO ground-state factors: exp(-x²/2) as a single Gaussian edge
    - Combined with HermiteBasis via MultKAN mult nodes to get ψ_n(x) = H_n(x)·exp(-x²/2)
    - Smooth functions on ℝ with localized support

Each Gaussian is parameterized by a learnable center μ_m and width σ_m:
    φ_m(x) = exp(-((x - μ_m) / σ_m)²)
"""

import math
from typing import Dict, Any, List, Optional

import mlx.core as mx

from .base import Basis, BasisConfig, softplus


class GaussianBasis(Basis):
    """Gaussian radial basis functions (RBFs).

    φ_m(x) = exp(-((x - μ_m) / σ_m)²)

    M Gaussians with learnable centers μ_m and widths σ_m > 0.

    Domain: ℝ (no domain mapping needed).

    Args:
        config: BasisConfig specifying M basis functions
        center_range: (lo, hi) range for initial uniform center spacing (default: (-3, 3))
        init_width: Initial width σ for all Gaussians (default: 1.0)
        learnable_centers: Whether centers μ_m are trainable (default: True)
        learnable_widths: Whether widths σ_m are trainable (default: True)

    Notes:
        - Widths are stored as log_widths so σ = softplus(log_widths) + ε > 0
        - For QHO ground state: a single [1,1] edge can learn exp(-x²/2) when
          center → 0 and σ → 1/√2
        - Multiply with HermiteBasis edges via MultKAN mult nodes for ψ_n(x)
    """

    name = "gaussian"

    def __init__(
        self,
        config: BasisConfig,
        center_range: tuple = (-3.0, 3.0),
        init_width: float = 1.0,
        learnable_centers: bool = True,
        learnable_widths: bool = True,
    ):
        # Gaussians are defined on all of ℝ — no domain mapping
        config_copy = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=(-float("inf"), float("inf")),
            normalize_method="none",
        )
        super().__init__(config_copy)
        self.center_range = center_range
        self.init_width = init_width
        self.learnable_centers = learnable_centers
        self.learnable_widths = learnable_widths

    def _init_extra_params(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
    ) -> Dict[str, mx.array]:
        """Initialize learnable centers and log-widths."""
        params = {}

        # Centers uniformly spaced in center_range
        centers_1d = mx.linspace(self.center_range[0], self.center_range[1], self.M)

        # log_widths inverse-softplus of init_width: softplus(log_w) = init_width
        # => log_w = log(exp(init_width) - 1)
        log_w_init = math.log(math.exp(self.init_width) - 1.0)

        if self.config.param_mode == "per_in":
            if self.learnable_centers:
                # (in_dim, M) — one set of centers per input dimension
                params["centers"] = mx.broadcast_to(
                    centers_1d[None, :], (in_dim, self.M)
                )
            if self.learnable_widths:
                params["log_widths"] = mx.full((in_dim, self.M), log_w_init)
        # per_edge mode not needed for current use cases; fall through returns {}

        return params

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate Gaussian basis functions.

        Args:
            x: Input of shape (batch, in_dim)
            params: Parameter dict with optional 'centers' (in_dim, M)
                    and 'log_widths' (in_dim, M)

        Returns:
            Gaussian values of shape (batch, in_dim, M)
        """
        x_norm = self._normalize(x, params)  # (B, I)

        centers = params.get("centers")      # (I, M) or None
        log_widths = params.get("log_widths")  # (I, M) or None

        if centers is None:
            # Fallback: fixed uniform centers
            centers = mx.linspace(self.center_range[0], self.center_range[1], self.M)
            # Broadcast to (1, M) for use below
            mu = centers[None, None, :]  # (1, 1, M) -> broadcasts over (B, I)
        else:
            mu = mx.expand_dims(centers, axis=0)  # (1, I, M)

        if log_widths is None:
            sg = mx.array(self.init_width)  # scalar
        else:
            sg = softplus(log_widths) + 1e-6   # (I, M), always positive
            sg = mx.expand_dims(sg, axis=0)     # (1, I, M)

        xb = mx.expand_dims(x_norm, axis=2)     # (B, I, 1)
        return mx.exp(-((xb - mu) / sg) ** 2)   # (B, I, M)

    @property
    def symbolic_priority(self) -> List[str]:
        """Prioritize Gaussian-related symbolic functions."""
        return ["gaussian", "exp", "x^2", "1"]

    def symbolic(
        self,
        coeffs: mx.array,
        params: Dict[str, Any],
        var: str = "x",
    ) -> str:
        """Generate symbolic formula string.

        Args:
            coeffs: Coefficients of shape (M,) for one edge
            params: Parameter dict with centers and log_widths
            var: Variable name to use

        Returns:
            Human-readable formula, e.g. "+0.9997*exp(-((x)/0.707)²)"
        """
        centers = params.get("centers")
        log_widths = params.get("log_widths")

        terms = []
        for m, c in enumerate(coeffs.tolist()):
            if abs(c) < 1e-6:
                continue

            mu = (
                float(centers[0, m]) if centers is not None else 0.0
            )
            if log_widths is not None:
                sg = float(softplus(log_widths[0:1, m : m + 1])[0, 0]) + 1e-6
            else:
                sg = self.init_width

            center_str = var if abs(mu) < 1e-4 else f"({var}{mu:+.3g})"
            terms.append(f"{c:+.4g}*exp(-(({center_str})/{sg:.3g})²)")

        return " ".join(terms) or "0"
