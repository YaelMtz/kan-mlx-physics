"""Hermite polynomial basis for quantum mechanics.

Hermite polynomials are the natural basis for:
    - Quantum harmonic oscillator eigenfunctions
    - Gaussian-weighted problems
    - Heat equation on unbounded domains
    - Schrödinger equation with harmonic potential

Two conventions exist:
    - Physicist's: H_n(x) satisfying H_{n+1} = 2x·H_n - 2n·H_{n-1}
    - Probabilist's: He_n(x) satisfying He_{n+1} = x·He_n - n·He_{n-1}

This module uses the physicist's convention with optional Gaussian weighting
to give the actual QHO wavefunctions: ψ_n(x) ∝ H_n(x)·exp(-x²/2)
"""

from typing import Dict, Any, Optional, List
import mlx.core as mx

from .base import Basis, BasisConfig


class HermiteBasis(Basis):
    """Hermite polynomials H_n(x) - physicist's convention.

    H_0(x) = 1
    H_1(x) = 2x
    H_n(x) = 2x·H_{n-1}(x) - 2(n-1)·H_{n-2}(x)

    Domain: ℝ (uses normalized x without domain mapping)

    Args:
        config: BasisConfig specifying M polynomials (degrees 0 to M-1)
        weighted: If True, multiply by exp(-x²/2) for QHO wavefunctions
        normalized: If True, normalize to have unit L² norm with Gaussian weight
    """

    name = "hermite"

    def __init__(
        self,
        config: BasisConfig,
        weighted: bool = False,
        normalized: bool = False,
    ):
        # Hermite polynomials are defined on all of R
        config_copy = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=(-float("inf"), float("inf")),
            normalize_method="none",
        )
        super().__init__(config_copy)

        self.weighted = weighted
        self.normalized = normalized

        # Precompute normalization constants if needed
        # ||H_n||² = 2^n · n! · √π under weight exp(-x²)
        # For weighted basis (ψ_n), ||ψ_n||² = 2^n · n! · √π
        if normalized:
            import math
            self._norm_factors = []
            for n in range(config.M):
                norm_sq = (2 ** n) * math.factorial(n) * math.sqrt(math.pi)
                self._norm_factors.append(1.0 / math.sqrt(norm_sq))

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate Hermite polynomials using recurrence.

        Args:
            x: Input of shape (batch, in_dim)
            params: Parameter dictionary

        Returns:
            Hermite values of shape (batch, in_dim, M)
        """
        # Apply affine normalization (shift and scale)
        x_norm = self._normalize(x, params)

        batch, in_dim = x_norm.shape

        # H_0 = 1
        H0 = mx.ones((batch, in_dim))

        if self.M == 1:
            result = H0[:, :, None]
        else:
            # H_1 = 2x
            H1 = 2 * x_norm

            polys = [H0[:, :, None], H1[:, :, None]]

            # Recurrence: H_{n+1} = 2x·H_n - 2n·H_{n-1}
            for n in range(1, self.M - 1):
                H2 = 2 * x_norm * H1 - 2 * n * H0
                polys.append(H2[:, :, None])
                H0, H1 = H1, H2

            result = mx.concatenate(polys[:self.M], axis=-1)

        # Apply Gaussian weight if requested
        if self.weighted:
            weight = mx.exp(-x_norm[:, :, None] ** 2 / 2)
            result = result * weight

        # Apply normalization if requested
        if self.normalized:
            norm_factors = mx.array(self._norm_factors)[None, None, :]
            result = result * norm_factors

        return result

    @property
    def symbolic_priority(self) -> List[str]:
        """Prioritize quantum harmonic oscillator functions."""
        return ["gaussian", "psi_0", "psi_1", "psi_2", "H_0", "H_1", "H_2", "exp", "x^2"]

    def symbolic(
        self,
        coeffs: mx.array,
        params: Dict[str, Any],
        var: str = "x",
    ) -> str:
        """Generate symbolic formula."""
        terms = []
        weight_str = f"*exp(-{var}²/2)" if self.weighted else ""

        for n, c in enumerate(coeffs.tolist()):
            if abs(c) < 1e-6:
                continue

            if self.normalized:
                c_display = c  # Already includes normalization
            else:
                c_display = c

            if n == 0:
                terms.append(f"{c_display:.4g}")
            elif n == 1:
                terms.append(f"{c_display:+.4g}*(2{var})")
            else:
                terms.append(f"{c_display:+.4g}*H_{n}({var})")

        if not terms:
            return "0"

        poly_part = " ".join(terms)
        if poly_part.startswith("+"):
            poly_part = poly_part[1:].strip()

        if self.weighted:
            return f"({poly_part})*exp(-{var}²/2)"
        return poly_part


class WeightedHermiteBasis(HermiteBasis):
    """Convenience alias for HermiteBasis with weighted=True.

    These are the actual quantum harmonic oscillator wavefunctions:
        ψ_n(x) ∝ H_n(x) · exp(-x²/2)
    """

    name = "weighted_hermite"

    def __init__(
        self,
        config: BasisConfig,
        normalized: bool = True,
    ):
        super().__init__(config, weighted=True, normalized=normalized)


class ProbabilistHermiteBasis(Basis):
    """Probabilist's Hermite polynomials He_n(x).

    He_0(x) = 1
    He_1(x) = x
    He_n(x) = x·He_{n-1}(x) - (n-1)·He_{n-2}(x)

    These are orthogonal under the standard Gaussian weight exp(-x²/2),
    which makes them natural for probability theory and statistics.
    """

    name = "probabilist_hermite"

    def __init__(
        self,
        config: BasisConfig,
        weighted: bool = False,
    ):
        config_copy = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=(-float("inf"), float("inf")),
            normalize_method="none",
        )
        super().__init__(config_copy)
        self.weighted = weighted

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate probabilist's Hermite polynomials."""
        x_norm = self._normalize(x, params)
        batch, in_dim = x_norm.shape

        # He_0 = 1
        He0 = mx.ones((batch, in_dim))

        if self.M == 1:
            result = He0[:, :, None]
        else:
            # He_1 = x
            He1 = x_norm

            polys = [He0[:, :, None], He1[:, :, None]]

            # Recurrence: He_{n+1} = x·He_n - n·He_{n-1}
            for n in range(1, self.M - 1):
                He2 = x_norm * He1 - n * He0
                polys.append(He2[:, :, None])
                He0, He1 = He1, He2

            result = mx.concatenate(polys[:self.M], axis=-1)

        if self.weighted:
            weight = mx.exp(-x_norm[:, :, None] ** 2 / 2)
            result = result * weight

        return result

    @property
    def symbolic_priority(self) -> List[str]:
        return ["gaussian", "exp", "x", "x^2", "x^3", "He_0", "He_1", "He_2"]
