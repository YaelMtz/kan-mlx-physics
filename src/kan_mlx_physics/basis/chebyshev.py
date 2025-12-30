"""Chebyshev polynomial basis for spectral methods.

Chebyshev polynomials are ideal for:
    - Spectral methods on bounded intervals
    - High-accuracy polynomial approximation
    - Functions with boundary layers
    - Numerical PDEs on [-1, 1]

Properties:
    - Orthogonal on [-1, 1] with weight 1/√(1-x²)
    - Minimax property: best polynomial approximation
    - Recurrence: T_{n+1}(x) = 2x·T_n(x) - T_{n-1}(x)
"""

from typing import Dict, Any, Optional, List
import mlx.core as mx

from .base import Basis, BasisConfig


class ChebyshevBasis(Basis):
    """Chebyshev polynomials T_n(x) of the first kind.

    T_0(x) = 1
    T_1(x) = x
    T_n(x) = 2x·T_{n-1}(x) - T_{n-2}(x)

    Domain: [-1, 1] (uses tanh mapping for inputs outside)

    Args:
        config: BasisConfig specifying M polynomials (degrees 0 to M-1)
    """

    name = "chebyshev"

    def __init__(self, config: BasisConfig):
        # Override domain to [-1, 1] with tanh mapping
        config_copy = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=(-1.0, 1.0),
            normalize_method="tanh",  # Smooth mapping to [-1, 1]
        )
        super().__init__(config_copy)

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate Chebyshev polynomials using recurrence.

        Args:
            x: Input of shape (batch, in_dim)
            params: Parameter dictionary

        Returns:
            Chebyshev values of shape (batch, in_dim, M)
        """
        # Normalize and map to [-1, 1]
        x_norm = self._normalize(x, params)
        x_mapped = self._map_to_domain(x_norm)

        batch, in_dim = x_mapped.shape

        # T_0 = 1
        T0 = mx.ones((batch, in_dim))

        if self.M == 1:
            return T0[:, :, None]

        # T_1 = x
        T1 = x_mapped

        polys = [T0[:, :, None], T1[:, :, None]]

        # Recurrence: T_{n+1} = 2x·T_n - T_{n-1}
        for n in range(1, self.M - 1):
            T2 = 2 * x_mapped * T1 - T0
            polys.append(T2[:, :, None])
            T0, T1 = T1, T2

        return mx.concatenate(polys[:self.M], axis=-1)

    @property
    def symbolic_priority(self) -> List[str]:
        """Prioritize polynomial functions."""
        return ["x", "x^2", "x^3", "x^4", "1", "T_0", "T_1", "T_2", "T_3"]

    def symbolic(
        self,
        coeffs: mx.array,
        params: Dict[str, Any],
        var: str = "x",
    ) -> str:
        """Generate symbolic formula in Chebyshev form."""
        terms = []
        for n, c in enumerate(coeffs.tolist()):
            if abs(c) < 1e-6:
                continue
            if n == 0:
                terms.append(f"{c:.4g}")
            elif n == 1:
                terms.append(f"{c:+.4g}*{var}")
            else:
                terms.append(f"{c:+.4g}*T_{n}({var})")

        if not terms:
            return "0"

        result = " ".join(terms)
        if result.startswith("+"):
            result = result[1:].strip()
        return result


class ChebyshevSecondKindBasis(Basis):
    """Chebyshev polynomials U_n(x) of the second kind.

    U_0(x) = 1
    U_1(x) = 2x
    U_n(x) = 2x·U_{n-1}(x) - U_{n-2}(x)

    These satisfy: sin((n+1)θ) = sin(θ)·U_n(cos(θ))

    Useful for:
        - Derivative representations (d/dx T_n = n·U_{n-1})
        - Certain integral equations
    """

    name = "chebyshev_U"

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
        """Evaluate Chebyshev U polynomials."""
        x_norm = self._normalize(x, params)
        x_mapped = self._map_to_domain(x_norm)

        batch, in_dim = x_mapped.shape

        # U_0 = 1
        U0 = mx.ones((batch, in_dim))

        if self.M == 1:
            return U0[:, :, None]

        # U_1 = 2x
        U1 = 2 * x_mapped

        polys = [U0[:, :, None], U1[:, :, None]]

        # Recurrence: U_{n+1} = 2x·U_n - U_{n-1}
        for n in range(1, self.M - 1):
            U2 = 2 * x_mapped * U1 - U0
            polys.append(U2[:, :, None])
            U0, U1 = U1, U2

        return mx.concatenate(polys[:self.M], axis=-1)

    @property
    def symbolic_priority(self) -> List[str]:
        return ["x", "x^2", "x^3", "1", "U_0", "U_1", "U_2"]
