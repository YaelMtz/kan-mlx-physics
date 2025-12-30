"""Laguerre polynomial basis for radial problems.

Laguerre polynomials are the natural basis for:
    - Hydrogen atom radial wavefunctions
    - Radial Schrödinger equation
    - Problems on semi-infinite domains [0, ∞)
    - Exponentially decaying solutions

Generalized Laguerre polynomials L_n^α(x) satisfy:
    - Orthogonality on [0, ∞) with weight x^α · exp(-x)
    - Recurrence: (n+1)L_{n+1}^α = (2n+1+α-x)L_n^α - (n+α)L_{n-1}^α

The hydrogen atom radial functions are:
    R_{nl}(r) ∝ r^l · L_{n-l-1}^{2l+1}(2r/na₀) · exp(-r/na₀)
"""

from typing import Dict, Any, Optional, List
import mlx.core as mx

from .base import Basis, BasisConfig


class LaguerreBasis(Basis):
    """Generalized Laguerre polynomials L_n^α(x).

    L_0^α(x) = 1
    L_1^α(x) = 1 + α - x
    (n+1)L_{n+1}^α = (2n+1+α-x)L_n^α - (n+α)L_{n-1}^α

    Domain: [0, ∞) (uses sigmoid mapping: R -> (0, ∞))

    Args:
        config: BasisConfig specifying M polynomials
        alpha: Generalized Laguerre parameter (default 0)
        weighted: If True, multiply by x^(α/2) · exp(-x/2)
    """

    name = "laguerre"

    def __init__(
        self,
        config: BasisConfig,
        alpha: float = 0.0,
        weighted: bool = False,
    ):
        # Laguerre polynomials on [0, ∞)
        # Use sigmoid to map R -> (0, inf) smoothly
        config_copy = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=(0.0, float("inf")),
            normalize_method="none",  # We handle this specially
        )
        super().__init__(config_copy)

        self.alpha = alpha
        self.weighted = weighted

    def _map_to_positive(self, x: mx.array) -> mx.array:
        """Map normalized x to [0, ∞) using softplus.

        softplus(x) = log(1 + exp(x)) is smooth and positive,
        better than hard clipping.
        """
        from .base import softplus
        return softplus(x, beta=1.0) + 1e-6

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate Laguerre polynomials using recurrence.

        Args:
            x: Input of shape (batch, in_dim)
            params: Parameter dictionary

        Returns:
            Laguerre values of shape (batch, in_dim, M)
        """
        # Apply affine normalization
        x_norm = self._normalize(x, params)

        # Map to positive domain
        x_pos = self._map_to_positive(x_norm)

        batch, in_dim = x_pos.shape
        alpha = self.alpha

        # L_0^α = 1
        L0 = mx.ones((batch, in_dim))

        if self.M == 1:
            result = L0[:, :, None]
        else:
            # L_1^α = 1 + α - x
            L1 = 1 + alpha - x_pos

            polys = [L0[:, :, None], L1[:, :, None]]

            # Recurrence: (n+1)L_{n+1} = (2n+1+α-x)L_n - (n+α)L_{n-1}
            for n in range(1, self.M - 1):
                L2 = ((2 * n + 1 + alpha - x_pos) * L1 - (n + alpha) * L0) / (n + 1)
                polys.append(L2[:, :, None])
                L0, L1 = L1, L2

            result = mx.concatenate(polys[:self.M], axis=-1)

        # Apply weight if requested: x^(α/2) · exp(-x/2)
        if self.weighted:
            weight = (x_pos ** (self.alpha / 2))[:, :, None] * mx.exp(-x_pos[:, :, None] / 2)
            result = result * weight

        return result

    @property
    def symbolic_priority(self) -> List[str]:
        """Prioritize hydrogen-like radial functions."""
        return ["exp", "R_10", "R_20", "R_21", "L_0", "L_1", "L_2", "x", "x^2"]

    def symbolic(
        self,
        coeffs: mx.array,
        params: Dict[str, Any],
        var: str = "x",
    ) -> str:
        """Generate symbolic formula."""
        terms = []
        alpha_str = f"^{{{self.alpha}}}" if self.alpha != 0 else ""

        for n, c in enumerate(coeffs.tolist()):
            if abs(c) < 1e-6:
                continue
            if n == 0:
                terms.append(f"{c:.4g}")
            else:
                terms.append(f"{c:+.4g}*L_{n}{alpha_str}({var})")

        if not terms:
            return "0"

        poly_part = " ".join(terms)
        if poly_part.startswith("+"):
            poly_part = poly_part[1:].strip()

        if self.weighted:
            weight_str = f"{var}^{{{self.alpha/2:.2g}}}*" if self.alpha != 0 else ""
            return f"({poly_part})*{weight_str}exp(-{var}/2)"
        return poly_part


class AssociatedLaguerreBasis(LaguerreBasis):
    """Associated Laguerre polynomials (alias for generalized with integer α)."""

    name = "associated_laguerre"

    def __init__(
        self,
        config: BasisConfig,
        k: int = 0,
        weighted: bool = False,
    ):
        """
        Args:
            config: BasisConfig
            k: Association parameter (integer α)
            weighted: Apply exponential weight
        """
        super().__init__(config, alpha=float(k), weighted=weighted)
        self.k = k


class HydrogenRadialBasis(Basis):
    """Hydrogen atom radial basis functions R_{nl}(r).

    These are the exact radial wavefunctions for hydrogen:
        R_{nl}(r) = N_{nl} · (2r/n)^l · L_{n-l-1}^{2l+1}(2r/n) · exp(-r/n)

    where a₀ = 1 (atomic units) and N_{nl} is normalization.

    Args:
        config: BasisConfig (M = number of (n,l) pairs to include)
        max_n: Maximum principal quantum number
        a0: Bohr radius (default 1 for atomic units)
    """

    name = "hydrogen_radial"

    def __init__(
        self,
        config: BasisConfig,
        max_n: int = 3,
        a0: float = 1.0,
    ):
        config_copy = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=(0.0, float("inf")),
            normalize_method="none",
        )
        super().__init__(config_copy)

        self.max_n = max_n
        self.a0 = a0

        # Build list of (n, l) quantum numbers
        self.quantum_numbers = []
        for n in range(1, max_n + 1):
            for l in range(n):
                self.quantum_numbers.append((n, l))
                if len(self.quantum_numbers) >= config.M:
                    break
            if len(self.quantum_numbers) >= config.M:
                break

        # Adjust M to match available states
        self.M = min(config.M, len(self.quantum_numbers))

    def _map_to_positive(self, x: mx.array) -> mx.array:
        from .base import softplus
        return softplus(x, beta=1.0) + 1e-6

    def _laguerre_value(
        self,
        x: mx.array,
        n: int,
        alpha: float,
    ) -> mx.array:
        """Evaluate single Laguerre polynomial L_n^α(x)."""
        if n == 0:
            return mx.ones_like(x)

        L0 = mx.ones_like(x)
        L1 = 1 + alpha - x

        if n == 1:
            return L1

        for k in range(1, n):
            L2 = ((2 * k + 1 + alpha - x) * L1 - (k + alpha) * L0) / (k + 1)
            L0, L1 = L1, L2

        return L1

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate hydrogen radial functions."""
        x_norm = self._normalize(x, params)
        r = self._map_to_positive(x_norm)

        batch, in_dim = r.shape
        features_list = []

        for n, l in self.quantum_numbers[:self.M]:
            # Compute R_{nl}(r)
            rho = 2 * r / (n * self.a0)

            # L_{n-l-1}^{2l+1}(rho)
            lag_n = n - l - 1
            lag_alpha = 2 * l + 1
            L = self._laguerre_value(rho, lag_n, lag_alpha)

            # Full radial function (unnormalized)
            # R_{nl} ∝ rho^l · L · exp(-rho/2)
            R = (rho ** l) * L * mx.exp(-rho / 2)

            features_list.append(R[:, :, None])

        return mx.concatenate(features_list, axis=-1)

    @property
    def symbolic_priority(self) -> List[str]:
        return ["R_10", "R_20", "R_21", "R_30", "R_31", "R_32", "exp", "x", "x^2"]

    def symbolic(
        self,
        coeffs: mx.array,
        params: Dict[str, Any],
        var: str = "r",
    ) -> str:
        terms = []
        for idx, c in enumerate(coeffs.tolist()):
            if idx >= len(self.quantum_numbers):
                break
            if abs(c) < 1e-6:
                continue
            n, l = self.quantum_numbers[idx]
            terms.append(f"{c:+.4g}*R_{{{n}{l}}}({var})")

        if not terms:
            return "0"

        result = " ".join(terms)
        if result.startswith("+"):
            result = result[1:].strip()
        return result
