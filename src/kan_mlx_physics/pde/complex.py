"""Complex wavefunction support for quantum PDEs.

Provides utilities for handling complex-valued wavefunctions Ψ = ψ_R + iψ_I
using a split real/imaginary representation for MLX compatibility.

Example:
    from kan_mlx_physics.pde.complex import ComplexWavefunction

    # Model outputs 2 components: [Re(Ψ), Im(Ψ)]
    model = MultKAN(width=[1, 20, 20, 2], ...)
    psi = ComplexWavefunction(model)

    # Get amplitude and phase
    amp = psi.amplitude(x)   # |Ψ|
    phi = psi.phase(x)       # arg(Ψ)

    # Probability current
    j = psi.probability_current(x, hbar=1.0)
"""

import mlx.core as mx
from typing import Tuple, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..multkan import MultKAN


class ComplexWavefunction:
    """Wrapper for complex-valued KAN wavefunction output.

    The KAN model outputs 2 components: [ψ_R, ψ_I] representing
    the real and imaginary parts of Ψ = ψ_R + iψ_I.

    This class provides convenient methods for computing physical
    quantities like amplitude, phase, and probability current.

    Args:
        model: MultKAN model with 2-component output
        hbar: Reduced Planck constant (default: 1.0)
        mass: Particle mass for probability current (default: 1.0)

    Attributes:
        model: The underlying KAN model
        hbar: Planck constant
        mass: Particle mass
    """

    def __init__(
        self,
        model: "MultKAN",
        hbar: float = 1.0,
        mass: float = 1.0,
    ):
        self.model = model
        self.hbar = hbar
        self.mass = mass

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Evaluate wavefunction at points x.

        Args:
            x: Input coordinates of shape (batch, dim)

        Returns:
            Tuple of (psi_real, psi_imag), each of shape (batch,)
        """
        out = self.model(x)
        if out.shape[-1] != 2:
            raise ValueError(
                f"Model must output 2 components for complex wavefunction, "
                f"got shape {out.shape}"
            )
        return out[:, 0], out[:, 1]

    def real(self, x: mx.array) -> mx.array:
        """Get real part Re(Ψ).

        Args:
            x: Input coordinates of shape (batch, dim)

        Returns:
            Real part of shape (batch,)
        """
        psi_R, _ = self(x)
        return psi_R

    def imag(self, x: mx.array) -> mx.array:
        """Get imaginary part Im(Ψ).

        Args:
            x: Input coordinates of shape (batch, dim)

        Returns:
            Imaginary part of shape (batch,)
        """
        _, psi_I = self(x)
        return psi_I

    def amplitude(self, x: mx.array) -> mx.array:
        """Compute amplitude |Ψ| = √(ψ_R² + ψ_I²).

        Args:
            x: Input coordinates of shape (batch, dim)

        Returns:
            Amplitude of shape (batch,)
        """
        psi_R, psi_I = self(x)
        return mx.sqrt(psi_R**2 + psi_I**2 + 1e-12)  # Small epsilon for stability

    def phase(self, x: mx.array) -> mx.array:
        """Compute phase arg(Ψ) = arctan2(ψ_I, ψ_R).

        Args:
            x: Input coordinates of shape (batch, dim)

        Returns:
            Phase in radians of shape (batch,), range [-π, π]
        """
        psi_R, psi_I = self(x)
        return mx.arctan2(psi_I, psi_R)

    def probability_density(self, x: mx.array) -> mx.array:
        """Compute probability density |Ψ|² = ψ_R² + ψ_I².

        Args:
            x: Input coordinates of shape (batch, dim)

        Returns:
            Probability density of shape (batch,)
        """
        psi_R, psi_I = self(x)
        return psi_R**2 + psi_I**2

    def probability_current(
        self,
        x: mx.array,
        h: float = 1e-4,
    ) -> mx.array:
        """Compute probability current j = (ℏ/m) Im(Ψ* ∇Ψ).

        For 1D: j = (ℏ/m) Im(Ψ* dΨ/dx)
              = (ℏ/m) (ψ_R ∂ψ_I/∂x - ψ_I ∂ψ_R/∂x)

        Args:
            x: Input coordinates of shape (batch, dim)
            h: Finite difference step size

        Returns:
            Probability current of shape (batch, dim)
        """
        psi_R, psi_I = self(x)
        batch_size, dim = x.shape

        j = mx.zeros((batch_size, dim))

        for d in range(dim):
            # Finite difference for gradients
            x_plus = x.at[:, d].add(h)
            x_minus = x.at[:, d].add(-h)

            psi_R_plus, psi_I_plus = self(x_plus)
            psi_R_minus, psi_I_minus = self(x_minus)

            # Central difference
            dpsi_R_dx = (psi_R_plus - psi_R_minus) / (2 * h)
            dpsi_I_dx = (psi_I_plus - psi_I_minus) / (2 * h)

            # j_d = (hbar/m) * Im(Ψ* ∂Ψ/∂x_d)
            #     = (hbar/m) * (ψ_R ∂ψ_I/∂x_d - ψ_I ∂ψ_R/∂x_d)
            j_d = (self.hbar / self.mass) * (psi_R * dpsi_I_dx - psi_I * dpsi_R_dx)
            j = j.at[:, d].set(j_d)

        return j

    def conjugate(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Compute complex conjugate Ψ* = ψ_R - iψ_I.

        Args:
            x: Input coordinates of shape (batch, dim)

        Returns:
            Tuple of (psi_real, -psi_imag)
        """
        psi_R, psi_I = self(x)
        return psi_R, -psi_I

    def inner_product(
        self,
        other: "ComplexWavefunction",
        x: mx.array,
        weights: Optional[mx.array] = None,
    ) -> Tuple[float, float]:
        """Compute inner product ⟨self|other⟩ = ∫ Ψ₁* Ψ₂ dx.

        Uses numerical integration over sample points.

        Args:
            other: Another ComplexWavefunction
            x: Sample points of shape (n_samples, dim)
            weights: Integration weights (default: uniform)

        Returns:
            Tuple of (real_part, imag_part) of ⟨self|other⟩
        """
        psi1_R, psi1_I = self(x)
        psi2_R, psi2_I = other(x)

        if weights is None:
            weights = mx.ones(x.shape[0]) / x.shape[0]

        # ⟨Ψ₁|Ψ₂⟩ = ∫ Ψ₁* Ψ₂ dx
        #         = ∫ (ψ₁_R - iψ₁_I)(ψ₂_R + iψ₂_I) dx
        #         = ∫ [(ψ₁_R ψ₂_R + ψ₁_I ψ₂_I) + i(ψ₁_R ψ₂_I - ψ₁_I ψ₂_R)] dx
        real_part = mx.sum(weights * (psi1_R * psi2_R + psi1_I * psi2_I))
        imag_part = mx.sum(weights * (psi1_R * psi2_I - psi1_I * psi2_R))

        return float(real_part), float(imag_part)

    def norm(self, x: mx.array, weights: Optional[mx.array] = None) -> float:
        """Compute L² norm √⟨Ψ|Ψ⟩ = √(∫|Ψ|² dx).

        Args:
            x: Sample points of shape (n_samples, dim)
            weights: Integration weights (default: uniform)

        Returns:
            L² norm as float
        """
        real, imag = self.inner_product(self, x, weights)
        return float(mx.sqrt(mx.array(real)))  # Imag part should be ~0

    def normalize(
        self,
        x: mx.array,
        weights: Optional[mx.array] = None,
    ) -> float:
        """Compute normalization factor for the wavefunction.

        Note: This doesn't modify the model, just returns the factor.

        Args:
            x: Sample points
            weights: Integration weights

        Returns:
            Normalization factor N such that N*Ψ has unit norm
        """
        current_norm = self.norm(x, weights)
        if current_norm < 1e-12:
            return 1.0
        return 1.0 / current_norm


def complex_residual(
    problem_residual_fn: Callable,
    model: "MultKAN",
    x: mx.array,
) -> mx.array:
    """Compute PDE residual for complex wavefunction.

    For a real operator H, the equation HΨ = 0 becomes:
    - H ψ_R = 0  (real part)
    - H ψ_I = 0  (imaginary part)

    Args:
        problem_residual_fn: Function computing residual for real wavefunction
        model: KAN model with 2-component output
        x: Input coordinates

    Returns:
        Combined residual √(res_R² + res_I²)
    """
    out = model(x)
    psi_R, psi_I = out[:, 0:1], out[:, 1:2]

    # Create wrapper functions for each component
    class RealWrapper:
        def __call__(self, x):
            return model(x)[:, 0:1]

    class ImagWrapper:
        def __call__(self, x):
            return model(x)[:, 1:2]

    # Compute residual for each component
    res_R = problem_residual_fn(RealWrapper(), x)
    res_I = problem_residual_fn(ImagWrapper(), x)

    # Combined residual
    return mx.sqrt(res_R**2 + res_I**2 + 1e-12)


def complex_bc_loss(
    psi_R: mx.array,
    psi_I: mx.array,
    target_R: mx.array,
    target_I: mx.array,
) -> mx.array:
    """Compute boundary condition loss for complex wavefunction.

    Args:
        psi_R: Real part at boundary
        psi_I: Imaginary part at boundary
        target_R: Target real part
        target_I: Target imaginary part

    Returns:
        BC loss |Ψ - target|²
    """
    return (psi_R - target_R)**2 + (psi_I - target_I)**2


def wkb_phase_loss(
    model: "MultKAN",
    x: mx.array,
    S: Callable[[mx.array], mx.array],
    hbar: float = 1.0,
    sign: int = 1,
    h: float = 1e-4,
) -> mx.array:
    """Loss for WKB matching: Ψ ~ exp(±iS/ℏ).

    For WKB matching, we want:
    - ψ_R ≈ A cos(S/ℏ)
    - ψ_I ≈ ±A sin(S/ℏ)

    This gives: arg(Ψ) ≈ ±S/ℏ

    Args:
        model: KAN model with 2-component output
        x: Points at which to match WKB
        S: Classical action function S(x)
        hbar: Planck constant
        sign: +1 for outgoing, -1 for incoming wave
        h: Finite difference step

    Returns:
        Phase matching loss
    """
    out = model(x)
    psi_R, psi_I = out[:, 0], out[:, 1]

    # Computed phase
    computed_phase = mx.arctan2(psi_I, psi_R)

    # Target phase from WKB
    target_phase = sign * S(x) / hbar

    # Wrap phase difference to [-π, π]
    phase_diff = computed_phase - target_phase
    phase_diff = mx.arctan2(mx.sin(phase_diff), mx.cos(phase_diff))

    return mx.mean(phase_diff**2)
