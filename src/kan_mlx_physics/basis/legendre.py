"""Legendre polynomial basis for angular momentum and spherical problems.

Legendre polynomials are the natural basis for:
    - Angular momentum eigenfunctions
    - Spherical harmonics (as associated Legendre functions)
    - Problems with spherical symmetry
    - Multipole expansions

Properties:
    - Orthogonal on [-1, 1] with unit weight
    - P_l(1) = 1, P_l(-1) = (-1)^l
    - Recurrence: (l+1)P_{l+1} = (2l+1)x·P_l - l·P_{l-1}
"""

from typing import Dict, Any, Optional, List
import mlx.core as mx

from .base import Basis, BasisConfig


class LegendreBasis(Basis):
    """Legendre polynomials P_l(x).

    P_0(x) = 1
    P_1(x) = x
    (l+1)P_{l+1}(x) = (2l+1)x·P_l(x) - l·P_{l-1}(x)

    Domain: [-1, 1] (uses tanh mapping)

    Args:
        config: BasisConfig specifying M polynomials (degrees 0 to M-1)
    """

    name = "legendre"

    def __init__(self, config: BasisConfig):
        config_copy = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=(-1.0, 1.0),
            normalize_method="tanh",
        )
        super().__init__(config_copy)

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate Legendre polynomials using recurrence.

        Args:
            x: Input of shape (batch, in_dim)
            params: Parameter dictionary

        Returns:
            Legendre values of shape (batch, in_dim, M)
        """
        x_norm = self._normalize(x, params)
        x_mapped = self._map_to_domain(x_norm)

        batch, in_dim = x_mapped.shape

        # P_0 = 1
        P0 = mx.ones((batch, in_dim))

        if self.M == 1:
            return P0[:, :, None]

        # P_1 = x
        P1 = x_mapped

        polys = [P0[:, :, None], P1[:, :, None]]

        # Recurrence: (l+1)P_{l+1} = (2l+1)x·P_l - l·P_{l-1}
        for l in range(1, self.M - 1):
            P2 = ((2 * l + 1) * x_mapped * P1 - l * P0) / (l + 1)
            polys.append(P2[:, :, None])
            P0, P1 = P1, P2

        return mx.concatenate(polys[:self.M], axis=-1)

    @property
    def symbolic_priority(self) -> List[str]:
        """Prioritize polynomial and spherical functions."""
        return ["x", "x^2", "x^3", "P_0", "P_1", "P_2", "P_3", "1"]

    def symbolic(
        self,
        coeffs: mx.array,
        params: Dict[str, Any],
        var: str = "x",
    ) -> str:
        """Generate symbolic formula."""
        terms = []
        for l, c in enumerate(coeffs.tolist()):
            if abs(c) < 1e-6:
                continue
            if l == 0:
                terms.append(f"{c:.4g}")
            elif l == 1:
                terms.append(f"{c:+.4g}*{var}")
            else:
                terms.append(f"{c:+.4g}*P_{l}({var})")

        if not terms:
            return "0"

        result = " ".join(terms)
        if result.startswith("+"):
            result = result[1:].strip()
        return result


class AssociatedLegendreBasis(Basis):
    """Associated Legendre functions P_l^m(x).

    These are related to spherical harmonics:
        Y_l^m(θ, φ) ∝ P_l^m(cos θ) · exp(imφ)

    Computed via:
        P_l^m(x) = (-1)^m (1-x²)^{m/2} d^m/dx^m P_l(x)

    Args:
        config: BasisConfig (M determines max l)
        m: Azimuthal quantum number (0 ≤ m ≤ l)
    """

    name = "associated_legendre"

    def __init__(
        self,
        config: BasisConfig,
        m: int = 0,
    ):
        config_copy = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=(-1.0, 1.0),
            normalize_method="tanh",
        )
        super().__init__(config_copy)
        self.m = abs(m)

        # Adjust M: only l >= m contribute
        # So we have l = m, m+1, ..., m+M-1
        self.l_values = list(range(self.m, self.m + config.M))

    def _legendre_Pl(self, x: mx.array, l_max: int) -> List[mx.array]:
        """Compute all Legendre polynomials up to l_max."""
        batch, in_dim = x.shape

        P0 = mx.ones((batch, in_dim))
        if l_max == 0:
            return [P0]

        P1 = x
        polys = [P0, P1]

        for l in range(1, l_max):
            P2 = ((2 * l + 1) * x * P1 - l * P0) / (l + 1)
            polys.append(P2)
            P0, P1 = P1, P2

        return polys

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate associated Legendre functions.

        For m > 0, we use the relation:
            P_l^m(x) = (-1)^m (1-x²)^{m/2} d^m/dx^m P_l(x)

        For simplicity in MLX, we use numerical differentiation or
        the recurrence relations for P_l^m directly.
        """
        x_norm = self._normalize(x, params)
        x_mapped = self._map_to_domain(x_norm)

        batch, in_dim = x_mapped.shape

        if self.m == 0:
            # Just regular Legendre polynomials
            polys = self._legendre_Pl(x_mapped, max(self.l_values) + 1)
            selected = [polys[l][:, :, None] for l in self.l_values if l < len(polys)]
            return mx.concatenate(selected[:self.M], axis=-1)

        # For m > 0, use the recurrence for P_l^m
        # P_m^m(x) = (-1)^m (2m-1)!! (1-x²)^{m/2}
        # P_{m+1}^m(x) = x(2m+1) P_m^m(x)
        # (l-m+1)P_{l+1}^m = (2l+1)x P_l^m - (l+m) P_{l-1}^m

        # Compute (1 - x²)^{m/2}
        factor = mx.power(1 - x_mapped ** 2, self.m / 2)

        # Double factorial (2m-1)!!
        double_fact = 1.0
        for k in range(1, 2 * self.m, 2):
            double_fact *= k

        # P_m^m
        Pmm = ((-1) ** self.m) * double_fact * factor

        if self.M == 1 and self.l_values[0] == self.m:
            return Pmm[:, :, None]

        polys = []
        if self.m in self.l_values:
            polys.append(Pmm[:, :, None])

        if self.m + 1 <= max(self.l_values):
            # P_{m+1}^m
            Pm1m = x_mapped * (2 * self.m + 1) * Pmm
            if self.m + 1 in self.l_values:
                polys.append(Pm1m[:, :, None])

            # Recurrence for higher l
            P0, P1 = Pmm, Pm1m
            for l in range(self.m + 1, max(self.l_values)):
                P2 = ((2 * l + 1) * x_mapped * P1 - (l + self.m) * P0) / (l - self.m + 1)
                if l + 1 in self.l_values:
                    polys.append(P2[:, :, None])
                P0, P1 = P1, P2

        return mx.concatenate(polys[:self.M], axis=-1)

    @property
    def symbolic_priority(self) -> List[str]:
        return ["Y_lm", "P_lm", "x", "sqrt(1-x^2)", "sin", "cos"]


class SphericalHarmonicBasis(Basis):
    """Real spherical harmonics Y_l^m(θ, φ) basis.

    Uses real combinations:
        Y_l^0 = P_l(cos θ)
        Y_l^{m>0} ∝ P_l^m(cos θ) · cos(mφ)
        Y_l^{m<0} ∝ P_l^{|m|}(cos θ) · sin(|m|φ)

    Input x is interpreted as (θ, φ) for 2D input,
    or just cos(θ) for 1D input (axially symmetric case).

    Args:
        config: BasisConfig
        max_l: Maximum angular momentum quantum number
    """

    name = "spherical_harmonic"

    def __init__(
        self,
        config: BasisConfig,
        max_l: int = 2,
    ):
        super().__init__(config)
        self.max_l = max_l

        # Build (l, m) pairs up to M
        self.lm_pairs = []
        for l in range(max_l + 1):
            for m in range(-l, l + 1):
                self.lm_pairs.append((l, m))
                if len(self.lm_pairs) >= config.M:
                    break
            if len(self.lm_pairs) >= config.M:
                break

        self.M = min(config.M, len(self.lm_pairs))

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate spherical harmonics.

        For 1D input: x = cos(θ), returns P_l(x) (axially symmetric)
        For 2D input: x = (θ, φ), returns full Y_l^m

        This is a simplified implementation focusing on the 1D case.
        """
        batch = x.shape[0]

        if x.ndim == 1 or x.shape[1] == 1:
            # 1D case: x = cos(θ), only m=0 terms
            x_flat = x.reshape(batch, 1) if x.ndim == 1 else x

            x_norm = self._normalize(x_flat, params)
            x_mapped = self._map_to_domain(x_norm)

            # Compute Legendre polynomials
            polys = []
            P0 = mx.ones((batch, 1))
            polys.append(P0)

            if self.max_l > 0:
                P1 = x_mapped
                polys.append(P1)

                for l in range(1, self.max_l):
                    P2 = ((2 * l + 1) * x_mapped * P1 - l * P0) / (l + 1)
                    polys.append(P2)
                    P0, P1 = P1, P2

            # Select only the (l, 0) terms
            selected = []
            for l, m in self.lm_pairs[:self.M]:
                if m == 0 and l < len(polys):
                    selected.append(polys[l][:, :, None])
                else:
                    # For m != 0 in 1D, these are zero (need φ dependence)
                    selected.append(mx.zeros((batch, 1, 1)))

            return mx.concatenate(selected, axis=-1)

        # 2D case would need full implementation with θ, φ
        # For now, raise an error
        raise NotImplementedError("Full 2D spherical harmonics not yet implemented")

    @property
    def symbolic_priority(self) -> List[str]:
        return ["Y_00", "Y_10", "Y_11", "Y_20", "Y_21", "Y_22", "P_l", "cos", "sin"]
