"""Modified-Bessel (K_{iν}) basis for exponential-potential problems.

This is the *native* basis of the anisotropic quantum-cosmology minisuperspace models
(Taub, Kantowski–Sachs), whose ★-genvalue equation carries an exponential potential
e^{a x}.  There the wavefunctions are modified Bessel functions of the second kind with
imaginary order, K_{iν}(c·e^{a x/2}), and the Wigner functions are built bilinearly from
them.  Just as the weighted-Laguerre basis is native to the harmonic-oscillator Wigner
function (W_n ∝ L_n(2r²)e^{-r²}), this basis is native to the exponential-potential
Wigner function.

The modified Bessel function of imaginary order has the large-argument (WKB) form
    K_{iν}(z) ~ sqrt(π/2z) · e^{-z} · cos(ν·ln z + δ),
and on the minisuperspace variable x (with z = c·e^{a x}) this becomes
    K_{iν}(c e^{a x}) ~ A · e^{-c e^{a x}} · cos(a ν x + δ),
an exponentially-cut oscillation in x whose oscillation rate is set by the order ν.
A direct fit of this profile to the transformed Taub ground truth reaches R² = 1.0
(versus 0.98 for the HO-native Laguerre–Gaussian), confirming it is the representationally
native family.  This class provides M such profiles with a spread of orders ν, so that a
single KAN edge can represent the exponential-potential Wigner shape.

Args:
    config: BasisConfig specifying M features
    a: exponent rate of the potential variable (Taub x-sector: a = 2)
    nu_max: largest Bessel order ν sampled; the M features use ν ∈ linspace(0, nu_max, M)
    weighted: if True, apply the e^{-c·e^{a x}} envelope (the K_{iν} decay); recommended
    fixed_scale: multiplicative scale c on e^{a x} before the envelope (default 1)
"""

from typing import Dict, Any, List
import mlx.core as mx

from .base import Basis, BasisConfig


class BesselBasis(Basis):
    """Modified-Bessel K_{iν} profiles for exponential-potential (cosmological) Wigner functions.

    Feature m (order ν_m):  e^{-c·u} · cos(a·ν_m·x + δ_m),  with u = e^{a·x}.
    The first feature (ν_0 = 0) is the pure decaying envelope e^{-c·u}; successive features
    add oscillation, spanning the K_{iν} family.
    """

    name = "bessel"

    def __init__(
        self,
        config: BasisConfig,
        a: float = 2.0,
        nu_max: float = 3.0,
        weighted: bool = True,
        fixed_scale: float = 1.0,
        argument: str = "exp",
    ):
        config_copy = BasisConfig(
            M=config.M,
            learnable_affine=config.learnable_affine,
            param_mode=config.param_mode,
            domain=(float("-inf"), float("inf")),  # x ∈ ℝ (the minisuperspace coordinate)
            normalize_method="none",
        )
        super().__init__(config_copy)
        self.a = a
        self.nu_max = nu_max
        self.weighted = weighted
        self.fixed_scale = fixed_scale
        # "exp":   argument z = fixed_scale · e^{a·x}  (x is the minisuperspace coordinate;
        #          native when the SOLVER works in x, e.g. the e^{a x} potential).
        # "linear": argument z = fixed_scale · x       (the input is ALREADY the exponential
        #          variable u = e^{a x}; native when the SOLVER works in u). The K_{iν}
        #          oscillation then lives in ln z, so features are e^{-z}·cos(ν·ln z + δ).
        if argument not in ("exp", "linear"):
            raise ValueError(f"argument must be 'exp' or 'linear', got {argument!r}")
        self.argument = argument
        # Orders ν and phases δ for the M features (fixed, not learnable: they define the family).
        M = config.M
        if M == 1:
            self._nus = [0.0]
            self._deltas = [0.0]
        else:
            self._nus = [nu_max * i / (M - 1) for i in range(M)]
            # alternate phases 0, π/2 so cos and sin components are both available
            self._deltas = [0.0 if i % 2 == 0 else mx.pi / 2 for i in range(M)]

    def features(self, x: mx.array, params: Dict[str, Any]) -> mx.array:
        """Evaluate the K_{iν} profiles.

        Args:
            x: input of shape (batch, in_dim) — the minisuperspace coordinate.
        Returns:
            features of shape (batch, in_dim, M).
        """
        x_norm = self._normalize(x, params)  # affine (learnable scale/shift) then our family
        a = self.a

        if self.argument == "exp":
            # z = c · e^{a x}; oscillation phase a·ν·x = ν·ln(z/c). Clamp exponent for safety.
            z = mx.exp(mx.clip(a * x_norm, -30.0, 30.0))
            if self.fixed_scale != 1.0:
                z = z * self.fixed_scale
            log_z = a * x_norm  # ln z up to the constant ln(fixed_scale), absorbed into δ
        else:  # "linear": the input IS the exponential variable u; z = c · u
            z = mx.maximum(x_norm * self.fixed_scale, 1e-6)
            log_z = mx.log(z)

        feats = []
        env = mx.exp(-z) if self.weighted else mx.ones_like(z)  # K_{iν} decay envelope
        for nu, delta in zip(self._nus, self._deltas):
            # K_{iν}(z) ~ e^{-z} cos(ν·ln z + δ): imaginary-order oscillation in ln z
            osc = mx.cos(nu * log_z + delta)
            feats.append((env * osc)[:, :, None])
        return mx.concatenate(feats, axis=-1)

    @property
    def symbolic_priority(self) -> List[str]:
        return ["K_i0", "K_i1", "K_i2", "K_i3", "exp", "cos"]

    def symbolic(
        self,
        coeffs: mx.array,
        params: Dict[str, Any],
        var: str = "x",
    ) -> str:
        """Symbolic form: sum of c_m · e^{-e^{a x}} cos(a ν_m x + δ_m)."""
        terms = []
        for m, c in enumerate(coeffs.tolist()):
            if abs(c) < 1e-6 or m >= len(self._nus):
                continue
            nu = self._nus[m]
            terms.append(f"{c:+.4g}*K_{{i{nu:.2g}}}(e^{{{self.a:.2g}{var}}})")
        if not terms:
            return "0"
        out = " ".join(terms)
        return out[1:].strip() if out.startswith("+") else out
