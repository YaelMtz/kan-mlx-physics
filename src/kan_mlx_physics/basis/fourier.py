"""Fourier basis implementation for periodic and oscillatory functions.

Fourier basis is ideal for:
    - Periodic functions
    - Oscillatory behavior in PDEs
    - Eigenfunctions of the Laplacian
    - Wave equations
"""

from typing import Dict, Any, Optional, List
import mlx.core as mx

from .base import Basis, BasisConfig, softplus


class FourierBasis(Basis):
    """Fourier basis: [1, cos(ωx), sin(ωx), cos(2ωx), sin(2ωx), ...].

    Features are ordered as:
        M=1: [1]
        M=2: [1, cos(ωx)]
        M=3: [1, cos(ωx), sin(ωx)]
        M=4: [1, cos(ωx), sin(ωx), cos(2ωx)]
        M=5: [1, cos(ωx), sin(ωx), cos(2ωx), sin(2ωx)]
        ...

    This ensures exact M features regardless of odd/even M.

    Args:
        config: BasisConfig specifying M and other options
        learnable_freq: If True, the frequency scale ω is learnable
        base_freq: Initial frequency (default 1.0 for period 2π)
    """

    name = "fourier"

    def __init__(
        self,
        config: BasisConfig,
        learnable_freq: bool = True,
        base_freq: float = 1.0,
    ):
        # Fourier doesn't need domain normalization - it works on all of R
        config_copy = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=(-float("inf"), float("inf")),
            normalize_method="none",
        )
        super().__init__(config_copy)

        self.learnable_freq = learnable_freq
        self.base_freq = base_freq

        # Number of harmonics: we have 1 (constant) + 2*K features
        # So K = (M - 1) // 2, and we may have an extra cos if M is even
        self.K = (config.M - 1) // 2

    def _init_extra_params(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
    ) -> Dict[str, mx.array]:
        """Initialize frequency parameter if learnable."""
        params = {}
        if self.learnable_freq:
            # omega_raw: softplus(omega_raw) + eps -> positive frequency
            # Initialize so softplus(0) * base_freq ≈ base_freq * 0.693
            # To get base_freq, we solve softplus(x) = base_freq
            # For simplicity, initialize at 0 and let base_freq handle it
            if self.config.param_mode == "per_in":
                params["omega_raw"] = mx.zeros((in_dim,))
            else:  # per_edge
                if out_dim is None:
                    raise ValueError("out_dim required for per_edge")
                params["omega_raw"] = mx.zeros((in_dim, out_dim))
        return params

    def _get_omega(self, params: Dict[str, Any]) -> mx.array:
        """Get positive frequency from params."""
        if not self.learnable_freq:
            return mx.array(self.base_freq)

        omega_raw = params.get("omega_raw")
        if omega_raw is None:
            return mx.array(self.base_freq)

        # softplus ensures positive, multiply by base_freq for scaling
        return softplus(omega_raw) * self.base_freq + 1e-6

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate Fourier basis functions.

        Args:
            x: Input of shape (batch, in_dim)
            params: Parameter dictionary

        Returns:
            Basis values of shape (batch, in_dim, M)
        """
        # Apply affine normalization
        x_norm = self._normalize(x, params)  # (B, I)

        batch, in_dim = x_norm.shape
        omega = self._get_omega(params)

        # Handle omega shape for broadcasting
        if omega.ndim == 0:
            omega = mx.broadcast_to(omega, (in_dim,))
        elif omega.ndim == 2:
            # per_edge mode - average over outputs for now
            # (features don't depend on output dimension)
            omega = mx.mean(omega, axis=1)

        # Build features list
        features_list = [mx.ones((batch, in_dim, 1))]  # Constant term

        # Add cos/sin pairs up to K harmonics
        for k in range(1, self.K + 1):
            # arg: (B, I) * (I,) -> (B, I)
            arg = k * omega[None, :] * x_norm
            features_list.append(mx.cos(arg)[:, :, None])
            if len(features_list) < self.M:
                features_list.append(mx.sin(arg)[:, :, None])

        # Handle case where M is even: add one more cos
        if self.M > 1 + 2 * self.K:
            k = self.K + 1
            arg = k * omega[None, :] * x_norm
            features_list.append(mx.cos(arg)[:, :, None])

        # Concatenate and trim to exact M
        phi = mx.concatenate(features_list[:self.M], axis=-1)

        # Ensure exact shape
        assert phi.shape == (batch, in_dim, self.M), \
            f"Expected ({batch}, {in_dim}, {self.M}), got {phi.shape}"

        return phi

    @property
    def symbolic_priority(self) -> List[str]:
        """Prioritize trigonometric functions."""
        return ["sin", "cos", "sin_q", "cos_q", "1", "x"]

    def symbolic(
        self,
        coeffs: mx.array,
        params: Dict[str, Any],
        var: str = "x",
    ) -> str:
        """Generate symbolic formula.

        Args:
            coeffs: Coefficients of shape (M,)
            params: Parameter dictionary
            var: Variable name

        Returns:
            Formula string like "0.5 + 0.3*cos(x) - 0.2*sin(x) + ..."
        """
        omega = float(self._get_omega(params).mean())

        terms = []

        # Constant term
        c0 = float(coeffs[0])
        if abs(c0) > 1e-6:
            terms.append(f"{c0:.4g}")

        # Harmonics
        idx = 1
        for k in range(1, self.K + 1):
            if idx >= len(coeffs):
                break
            a = float(coeffs[idx])
            if abs(a) > 1e-6:
                freq_str = f"{k * omega:.4g}" if k * omega != 1 else ""
                terms.append(f"{a:+.4g}*cos({freq_str}*{var})")
            idx += 1

            if idx >= len(coeffs):
                break
            b = float(coeffs[idx])
            if abs(b) > 1e-6:
                freq_str = f"{k * omega:.4g}" if k * omega != 1 else ""
                terms.append(f"{b:+.4g}*sin({freq_str}*{var})")
            idx += 1

        # Extra cos if M is even
        if idx < len(coeffs):
            k = self.K + 1
            a = float(coeffs[idx])
            if abs(a) > 1e-6:
                freq_str = f"{k * omega:.4g}" if k * omega != 1 else ""
                terms.append(f"{a:+.4g}*cos({freq_str}*{var})")

        if not terms:
            return "0"

        result = " ".join(terms)
        # Clean up leading +
        if result.startswith("+"):
            result = result[1:].strip()
        return result


class ComplexFourierBasis(Basis):
    """Complex Fourier basis: exp(i·k·ω·x) for k = -K, ..., K.

    Uses real representation: [cos, sin] pairs for each frequency.
    Useful for quantum mechanics and wave equations where
    complex exponentials are natural.

    M determines the frequency range:
        M = 2K + 1 for frequencies -K to K
    """

    name = "complex_fourier"

    def __init__(
        self,
        config: BasisConfig,
        learnable_freq: bool = True,
        base_freq: float = 1.0,
    ):
        super().__init__(config)
        self.learnable_freq = learnable_freq
        self.base_freq = base_freq

        # M = 2K + 1, so K = (M - 1) // 2
        self.K = (config.M - 1) // 2

        # Actual M might need adjustment
        self.actual_M = 2 * self.K + 1
        if self.actual_M != config.M:
            # Adjust to ensure odd M
            self.actual_M = config.M

    def _init_extra_params(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
    ) -> Dict[str, mx.array]:
        params = {}
        if self.learnable_freq:
            if self.config.param_mode == "per_in":
                params["omega_raw"] = mx.zeros((in_dim,))
            else:
                if out_dim is None:
                    raise ValueError("out_dim required for per_edge")
                params["omega_raw"] = mx.zeros((in_dim, out_dim))
        return params

    def _get_omega(self, params: Dict[str, Any]) -> mx.array:
        if not self.learnable_freq:
            return mx.array(self.base_freq)
        omega_raw = params.get("omega_raw")
        if omega_raw is None:
            return mx.array(self.base_freq)
        return softplus(omega_raw) * self.base_freq + 1e-6

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate complex Fourier basis (real representation)."""
        x_norm = self._normalize(x, params)
        batch, in_dim = x_norm.shape
        omega = self._get_omega(params)

        if omega.ndim == 0:
            omega = mx.broadcast_to(omega, (in_dim,))
        elif omega.ndim == 2:
            omega = mx.mean(omega, axis=1)

        # Build symmetric frequency list
        features_list = []

        # Constant term (k=0)
        features_list.append(mx.ones((batch, in_dim, 1)))

        # Symmetric pairs: cos(kωx), sin(kωx) for k = 1, 2, ..., K
        for k in range(1, self.K + 1):
            arg = k * omega[None, :] * x_norm
            features_list.append(mx.cos(arg)[:, :, None])
            features_list.append(mx.sin(arg)[:, :, None])

        phi = mx.concatenate(features_list[:self.M], axis=-1)
        return phi

    @property
    def symbolic_priority(self) -> List[str]:
        return ["exp", "sin", "cos", "sin_q", "cos_q", "1"]
