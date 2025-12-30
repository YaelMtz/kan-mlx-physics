"""Physics-oriented symbolic functions for KAN-MLX-Physics.

This module provides symbolic functions relevant to:
- Quantum Mechanics (QM)
- Quantum Field Theory (QFT)
- Cosmology & General Relativity
- Algebraic Deformation Quantization
- Special Functions in Mathematical Physics

To use these functions, import and call register_physics_symbolic():
    from kan_mlx_physics.physics_symbolic import register_physics_symbolic
    register_physics_symbolic()
"""

import mlx.core as mx
import numpy as np
from scipy import special
from typing import Callable

from .symbolic import register_symbolic


def _np_to_mlx(fn_np: Callable) -> Callable:
    """Wrap a numpy function for MLX (via numpy conversion)."""
    def fn_mlx(x: mx.array) -> mx.array:
        x_np = np.array(x)
        result = fn_np(x_np)
        return mx.array(result.astype(np.float32))
    return fn_mlx


def register_physics_symbolic():
    """Register all physics-oriented symbolic functions."""

    # =========================================================================
    # QUANTUM MECHANICS
    # =========================================================================

    # --- Harmonic Oscillator ---
    # Hermite polynomials H_n(x) (first few)
    register_symbolic(
        "H_0", lambda x: mx.ones_like(x), lambda x: np.ones_like(x),
        "H_0(x)", "H_0(x)", complexity=1
    )
    register_symbolic(
        "H_1", lambda x: 2 * x, lambda x: 2 * x,
        "H_1(x) = 2x", "H_1(x)", complexity=2
    )
    register_symbolic(
        "H_2", lambda x: 4 * x**2 - 2, lambda x: 4 * x**2 - 2,
        "H_2(x) = 4x^2 - 2", "H_2(x)", complexity=3
    )
    register_symbolic(
        "H_3", lambda x: 8 * x**3 - 12 * x, lambda x: 8 * x**3 - 12 * x,
        "H_3(x) = 8x^3 - 12x", "H_3(x)", complexity=4
    )
    register_symbolic(
        "H_4", lambda x: 16 * x**4 - 48 * x**2 + 12,
        lambda x: 16 * x**4 - 48 * x**2 + 12,
        "H_4(x)", "H_4(x)", complexity=5
    )

    # Harmonic oscillator wavefunctions ψ_n(x) ∝ H_n(x) * exp(-x²/2)
    register_symbolic(
        "psi_0", lambda x: mx.exp(-x**2 / 2),
        lambda x: np.exp(-x**2 / 2),
        "\\psi_0(x) = e^{-x^2/2}", "psi_0(x) = e^(-x^2/2)", complexity=3
    )
    register_symbolic(
        "psi_1", lambda x: x * mx.exp(-x**2 / 2),
        lambda x: x * np.exp(-x**2 / 2),
        "\\psi_1(x) = x e^{-x^2/2}", "psi_1(x) = x e^(-x^2/2)", complexity=4
    )
    register_symbolic(
        "psi_2", lambda x: (2 * x**2 - 1) * mx.exp(-x**2 / 2),
        lambda x: (2 * x**2 - 1) * np.exp(-x**2 / 2),
        "\\psi_2(x)", "psi_2(x)", complexity=5
    )

    # --- Hydrogen Atom / Laguerre Polynomials ---
    register_symbolic(
        "L_0", lambda x: mx.ones_like(x), lambda x: np.ones_like(x),
        "L_0(x) = 1", "L_0(x)", complexity=1
    )
    register_symbolic(
        "L_1", lambda x: 1 - x, lambda x: 1 - x,
        "L_1(x) = 1 - x", "L_1(x)", complexity=2
    )
    register_symbolic(
        "L_2", lambda x: 1 - 2*x + x**2/2,
        lambda x: 1 - 2*x + x**2/2,
        "L_2(x) = 1 - 2x + \\frac{x^2}{2}", "L_2(x)", complexity=3
    )

    # Radial hydrogen wavefunction components R_nl
    register_symbolic(
        "R_10", lambda x: mx.exp(-x),
        lambda x: np.exp(-x),
        "R_{10} \\propto e^{-r}", "R_10 prop e^(-r)", complexity=2
    )
    register_symbolic(
        "R_20", lambda x: (1 - x/2) * mx.exp(-x/2),
        lambda x: (1 - x/2) * np.exp(-x/2),
        "R_{20} \\propto (1-r/2)e^{-r/2}", "R_20", complexity=4
    )
    register_symbolic(
        "R_21", lambda x: x * mx.exp(-x/2),
        lambda x: x * np.exp(-x/2),
        "R_{21} \\propto r e^{-r/2}", "R_21", complexity=3
    )

    # --- Spherical Harmonics (angular parts) ---
    # Y_l^m components (real forms for l=0,1,2)
    register_symbolic(
        "Y_00", lambda x: mx.ones_like(x) * 0.5 * np.sqrt(1/np.pi),
        lambda x: np.ones_like(x) * 0.5 * np.sqrt(1/np.pi),
        "Y_0^0", "Y_0^0", complexity=1
    )
    register_symbolic(
        "cos_theta", lambda x: mx.cos(x),
        lambda x: np.cos(x),
        "\\cos\\theta", "cos(theta)", complexity=2
    )
    register_symbolic(
        "sin_theta", lambda x: mx.sin(x),
        lambda x: np.sin(x),
        "\\sin\\theta", "sin(theta)", complexity=2
    )

    # --- Legendre Polynomials ---
    register_symbolic(
        "P_0", lambda x: mx.ones_like(x), lambda x: np.ones_like(x),
        "P_0(x) = 1", "P_0(x)", complexity=1
    )
    register_symbolic(
        "P_1", lambda x: x, lambda x: x,
        "P_1(x) = x", "P_1(x)", complexity=1
    )
    register_symbolic(
        "P_2", lambda x: (3*x**2 - 1) / 2,
        lambda x: (3*x**2 - 1) / 2,
        "P_2(x) = \\frac{3x^2 - 1}{2}", "P_2(x)", complexity=3
    )
    register_symbolic(
        "P_3", lambda x: (5*x**3 - 3*x) / 2,
        lambda x: (5*x**3 - 3*x) / 2,
        "P_3(x) = \\frac{5x^3 - 3x}{2}", "P_3(x)", complexity=4
    )
    register_symbolic(
        "P_4", lambda x: (35*x**4 - 30*x**2 + 3) / 8,
        lambda x: (35*x**4 - 30*x**2 + 3) / 8,
        "P_4(x)", "P_4(x)", complexity=5
    )

    # --- Chebyshev Polynomials (T_n) ---
    register_symbolic(
        "T_0", lambda x: mx.ones_like(x), lambda x: np.ones_like(x),
        "T_0(x) = 1", "T_0(x)", complexity=1
    )
    register_symbolic(
        "T_1", lambda x: x, lambda x: x,
        "T_1(x) = x", "T_1(x)", complexity=1
    )
    register_symbolic(
        "T_2", lambda x: 2*x**2 - 1,
        lambda x: 2*x**2 - 1,
        "T_2(x) = 2x^2 - 1", "T_2(x)", complexity=3
    )
    register_symbolic(
        "T_3", lambda x: 4*x**3 - 3*x,
        lambda x: 4*x**3 - 3*x,
        "T_3(x) = 4x^3 - 3x", "T_3(x)", complexity=4
    )

    # =========================================================================
    # SPECIAL FUNCTIONS (via scipy)
    # =========================================================================

    # --- Bessel Functions ---
    register_symbolic(
        "J_0", _np_to_mlx(special.j0), special.j0,
        "J_0(x)", "J_0(x)", complexity=3
    )
    register_symbolic(
        "J_1", _np_to_mlx(special.j1), special.j1,
        "J_1(x)", "J_1(x)", complexity=3
    )
    register_symbolic(
        "Y_0", _np_to_mlx(special.y0), special.y0,
        "Y_0(x)", "Y_0(x)", complexity=3
    )
    register_symbolic(
        "Y_1", _np_to_mlx(special.y1), special.y1,
        "Y_1(x)", "Y_1(x)", complexity=3
    )

    # Modified Bessel functions
    register_symbolic(
        "I_0", _np_to_mlx(special.i0), special.i0,
        "I_0(x)", "I_0(x)", complexity=3
    )
    register_symbolic(
        "I_1", _np_to_mlx(special.i1), special.i1,
        "I_1(x)", "I_1(x)", complexity=3
    )
    register_symbolic(
        "K_0", _np_to_mlx(special.k0), special.k0,
        "K_0(x)", "K_0(x)", complexity=3
    )
    register_symbolic(
        "K_1", _np_to_mlx(special.k1), special.k1,
        "K_1(x)", "K_1(x)", complexity=3
    )

    # Spherical Bessel functions
    def spherical_j0(x):
        return np.sinc(x / np.pi)  # sin(x)/x

    def spherical_j1(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(np.abs(x) < 1e-10, 0.0,
                             (np.sin(x) - x * np.cos(x)) / x**2)
        return result

    register_symbolic(
        "j_0", _np_to_mlx(spherical_j0), spherical_j0,
        "j_0(x) = \\frac{\\sin x}{x}", "j_0(x)", complexity=3
    )
    register_symbolic(
        "j_1", _np_to_mlx(spherical_j1), spherical_j1,
        "j_1(x)", "j_1(x)", complexity=4
    )

    # --- Error Function & Related ---
    register_symbolic(
        "erf", _np_to_mlx(special.erf), special.erf,
        "\\text{erf}(x)", "op(\"erf\")(x)", complexity=3
    )
    register_symbolic(
        "erfc", _np_to_mlx(special.erfc), special.erfc,
        "\\text{erfc}(x)", "op(\"erfc\")(x)", complexity=3
    )

    # --- Gamma & Related ---
    register_symbolic(
        "gamma", _np_to_mlx(special.gamma), special.gamma,
        "\\Gamma(x)", "Gamma(x)", complexity=4
    )
    register_symbolic(
        "loggamma", _np_to_mlx(special.gammaln), special.gammaln,
        "\\ln\\Gamma(x)", "ln(Gamma(x))", complexity=4
    )
    register_symbolic(
        "digamma", _np_to_mlx(special.digamma), special.digamma,
        "\\psi(x)", "psi(x)", complexity=4
    )

    # --- Airy Functions (QM tunneling, WKB) ---
    def airy_ai(x):
        return special.airy(x)[0]

    def airy_bi(x):
        return special.airy(x)[2]

    register_symbolic(
        "Ai", _np_to_mlx(airy_ai), airy_ai,
        "\\text{Ai}(x)", "op(\"Ai\")(x)", complexity=4
    )
    register_symbolic(
        "Bi", _np_to_mlx(airy_bi), airy_bi,
        "\\text{Bi}(x)", "op(\"Bi\")(x)", complexity=4
    )

    # =========================================================================
    # QUANTUM FIELD THEORY
    # =========================================================================

    # --- Propagator-like functions ---
    # Feynman propagator (Euclidean): 1/(p² + m²)
    register_symbolic(
        "propagator", lambda x: 1 / (x**2 + 1),
        lambda x: 1 / (x**2 + 1),
        "\\frac{1}{x^2 + 1}", "1/(x^2 + 1)", complexity=3
    )

    # Massive propagator with mass parameter
    register_symbolic(
        "yukawa", lambda x: mx.exp(-mx.abs(x)) / (mx.abs(x) + 1e-8),
        lambda x: np.exp(-np.abs(x)) / (np.abs(x) + 1e-8),
        "\\frac{e^{-|x|}}{|x|}", "e^(-|x|)/|x|", complexity=4
    )

    # Coulomb-like
    register_symbolic(
        "coulomb", lambda x: 1 / (mx.abs(x) + 1e-8),
        lambda x: 1 / (np.abs(x) + 1e-8),
        "\\frac{1}{|x|}", "1/|x|", complexity=2
    )

    # --- Thermal/Statistical QFT ---
    # Bose-Einstein distribution
    def bose_einstein(x):
        with np.errstate(over='ignore'):
            return 1 / (np.exp(x) - 1 + 1e-10)

    register_symbolic(
        "bose", _np_to_mlx(bose_einstein), bose_einstein,
        "\\frac{1}{e^x - 1}", "1/(e^x - 1)", complexity=4
    )

    # Fermi-Dirac distribution
    def fermi_dirac(x):
        return 1 / (np.exp(x) + 1)

    register_symbolic(
        "fermi", _np_to_mlx(fermi_dirac), fermi_dirac,
        "\\frac{1}{e^x + 1}", "1/(e^x + 1)", complexity=4
    )

    # Planck distribution (blackbody)
    def planck(x):
        with np.errstate(over='ignore', divide='ignore'):
            return np.where(np.abs(x) < 1e-10, 0.0,
                           x**3 / (np.exp(x) - 1 + 1e-10))

    register_symbolic(
        "planck", _np_to_mlx(planck), planck,
        "\\frac{x^3}{e^x - 1}", "x^3/(e^x - 1)", complexity=5
    )

    # --- Polylogarithms (appear in QFT loop calculations) ---
    def polylog2(x):
        # Li_2(x) = -∫_0^x ln(1-t)/t dt
        return special.spence(1 - x)  # spence is Li_2(1-x)

    register_symbolic(
        "Li_2", _np_to_mlx(lambda x: special.spence(1 - np.clip(x, -10, 0.999))),
        lambda x: special.spence(1 - np.clip(x, -10, 0.999)),
        "\\text{Li}_2(x)", "op(\"Li\")_2(x)", complexity=5
    )

    # =========================================================================
    # COSMOLOGY & GENERAL RELATIVITY
    # =========================================================================

    # --- Scale Factor Evolution ---
    # Matter-dominated: a(t) ∝ t^(2/3)
    register_symbolic(
        "a_matter", lambda x: mx.abs(x + 1e-8)**(2/3),
        lambda x: np.abs(x + 1e-8)**(2/3),
        "a \\propto t^{2/3}", "a prop t^(2/3)", complexity=3
    )

    # Radiation-dominated: a(t) ∝ t^(1/2)
    register_symbolic(
        "a_rad", lambda x: mx.sqrt(mx.abs(x) + 1e-8),
        lambda x: np.sqrt(np.abs(x) + 1e-8),
        "a \\propto t^{1/2}", "a prop t^(1/2)", complexity=2
    )

    # de Sitter (dark energy): a(t) ∝ exp(Ht)
    register_symbolic(
        "a_deSitter", lambda x: mx.exp(x),
        lambda x: np.exp(x),
        "a \\propto e^{Ht}", "a prop e^(H t)", complexity=2
    )

    # --- Redshift relations ---
    # Luminosity distance (flat ΛCDM approximation)
    register_symbolic(
        "D_L", lambda x: x * (1 + x),  # D_L ≈ (c/H_0) * z * (1 + z/2) simplified
        lambda x: x * (1 + x),
        "D_L \\propto z(1+z)", "D_L prop z(1+z)", complexity=3
    )

    # --- Metric Components ---
    # Schwarzschild factor
    register_symbolic(
        "schwarzschild", lambda x: 1 - 1/(mx.abs(x) + 1e-8),
        lambda x: 1 - 1/(np.abs(x) + 1e-8),
        "1 - \\frac{r_s}{r}", "1 - r_s/r", complexity=3
    )

    # =========================================================================
    # ALGEBRAIC DEFORMATION QUANTIZATION
    # =========================================================================

    # --- q-Deformed Functions ---
    # q-exponential: e_q(x) = (1 + (1-q)x)^(1/(1-q)) for q≠1
    # Here we use q=0 (Tsallis q-exponential limit cases)

    def q_exp_0(x):
        """q-exponential with q→0: max(0, 1+x)"""
        return np.maximum(0, 1 + x)

    register_symbolic(
        "q_exp_0", _np_to_mlx(q_exp_0), q_exp_0,
        "e_0(x) = \\max(0, 1+x)", "e_0(x)", complexity=2
    )

    # q-exponential with q=2 (relevant in some deformations)
    def q_exp_2(x):
        """q-exponential with q=2: 1/(1-x) for x<1"""
        return 1 / (1 - np.clip(x, -100, 0.99))

    register_symbolic(
        "q_exp_2", _np_to_mlx(q_exp_2), q_exp_2,
        "e_2(x) = \\frac{1}{1-x}", "e_2(x)", complexity=3
    )

    # Tsallis q-logarithm: ln_q(x) = (x^(1-q) - 1)/(1-q)
    def q_log_2(x):
        """q-logarithm with q=2: 1 - 1/x"""
        return 1 - 1/(np.abs(x) + 1e-8)

    register_symbolic(
        "q_log_2", _np_to_mlx(q_log_2), q_log_2,
        "\\ln_2(x) = 1 - 1/x", "ln_2(x)", complexity=3
    )

    # --- Quantum Groups / q-Numbers ---
    # [n]_q = (q^n - q^(-n))/(q - q^(-1)) → n as q→1
    # For q=exp(iπ/3), these give interesting algebraic structures

    # q-sine: sin_q(x) with q-deformation
    def q_sin(x, q=np.exp(0.1)):
        """Approximation of q-deformed sine"""
        return (np.exp(1j * x * q) - np.exp(-1j * x * q)).imag / (2 * (q - 1/q).real + 1e-8)

    # Simplified q-deformed functions for neural network fitting
    register_symbolic(
        "sin_q", lambda x: mx.sin(x) * (1 + 0.1 * mx.sin(x)**2),
        lambda x: np.sin(x) * (1 + 0.1 * np.sin(x)**2),
        "\\sin_q(x)", "sin_q(x)", complexity=4
    )

    register_symbolic(
        "cos_q", lambda x: mx.cos(x) * (1 + 0.1 * mx.cos(x)**2),
        lambda x: np.cos(x) * (1 + 0.1 * np.cos(x)**2),
        "\\cos_q(x)", "cos_q(x)", complexity=4
    )

    # --- Star Product Components (Moyal) ---
    # The Moyal star product: (f ⋆ g)(x) involves derivatives
    # Here we provide basis functions that appear in expansions

    register_symbolic(
        "moyal_1", lambda x: x * mx.exp(-x**2),
        lambda x: x * np.exp(-x**2),
        "x e^{-x^2}", "x e^(-x^2)", complexity=4
    )

    register_symbolic(
        "moyal_2", lambda x: (1 - 2*x**2) * mx.exp(-x**2),
        lambda x: (1 - 2*x**2) * np.exp(-x**2),
        "(1-2x^2)e^{-x^2}", "(1-2x^2)e^(-x^2)", complexity=5
    )

    # =========================================================================
    # ADDITIONAL PHYSICS FUNCTIONS
    # =========================================================================

    # --- Step Functions ---
    register_symbolic(
        "heaviside", lambda x: (mx.sign(x) + 1) / 2,
        lambda x: np.heaviside(x, 0.5),
        "\\Theta(x)", "Theta(x)", complexity=2
    )

    # Smooth approximation to step function
    register_symbolic(
        "smooth_step", lambda x: mx.sigmoid(10 * x),
        lambda x: 1 / (1 + np.exp(-10 * x)),
        "\\sigma(10x)", "sigma(10x)", complexity=2
    )

    # --- Sinc Function ---
    register_symbolic(
        "sinc", lambda x: mx.where(mx.abs(x) < 1e-8, mx.ones_like(x), mx.sin(x) / x),
        lambda x: np.sinc(x / np.pi),
        "\\text{sinc}(x) = \\frac{\\sin x}{x}", "op(\"sinc\")(x)", complexity=3
    )

    # --- Lorentzian (Breit-Wigner) ---
    register_symbolic(
        "lorentzian", lambda x: 1 / (1 + x**2),
        lambda x: 1 / (1 + x**2),
        "\\frac{1}{1 + x^2}", "1/(1 + x^2)", complexity=3
    )

    # --- Damped Oscillations ---
    register_symbolic(
        "damped_sin", lambda x: mx.exp(-mx.abs(x)) * mx.sin(x),
        lambda x: np.exp(-np.abs(x)) * np.sin(x),
        "e^{-|x|}\\sin(x)", "e^(-|x|) sin(x)", complexity=4
    )

    register_symbolic(
        "damped_cos", lambda x: mx.exp(-mx.abs(x)) * mx.cos(x),
        lambda x: np.exp(-np.abs(x)) * np.cos(x),
        "e^{-|x|}\\cos(x)", "e^(-|x|) cos(x)", complexity=4
    )

    # --- Wave Packets ---
    register_symbolic(
        "wave_packet", lambda x: mx.exp(-x**2) * mx.cos(5*x),
        lambda x: np.exp(-x**2) * np.cos(5*x),
        "e^{-x^2}\\cos(5x)", "e^(-x^2) cos(5x)", complexity=5
    )

    # --- Power Laws ---
    register_symbolic(
        "x^1.5", lambda x: mx.abs(x + 1e-8)**1.5 * mx.sign(x + 1e-8),
        lambda x: np.abs(x + 1e-8)**1.5 * np.sign(x + 1e-8),
        "x^{3/2}", "x^(3/2)", complexity=3
    )

    register_symbolic(
        "x^2.5", lambda x: mx.abs(x + 1e-8)**2.5 * mx.sign(x + 1e-8),
        lambda x: np.abs(x + 1e-8)**2.5 * np.sign(x + 1e-8),
        "x^{5/2}", "x^(5/2)", complexity=3
    )

    register_symbolic(
        "x^-0.5", lambda x: 1 / mx.sqrt(mx.abs(x) + 1e-8),
        lambda x: 1 / np.sqrt(np.abs(x) + 1e-8),
        "x^{-1/2}", "x^(-1/2)", complexity=3
    )

    register_symbolic(
        "x^-1.5", lambda x: 1 / (mx.abs(x + 1e-8)**1.5),
        lambda x: 1 / (np.abs(x + 1e-8)**1.5),
        "x^{-3/2}", "x^(-3/2)", complexity=3
    )

    # --- Elliptic Integrals (appear in GR and field theory) ---
    register_symbolic(
        "ellipK", _np_to_mlx(lambda x: special.ellipk(np.clip(x, -0.99, 0.99))),
        lambda x: special.ellipk(np.clip(x, -0.99, 0.99)),
        "K(x)", "K(x)", complexity=5
    )

    register_symbolic(
        "ellipE", _np_to_mlx(lambda x: special.ellipe(np.clip(x, -0.99, 0.99))),
        lambda x: special.ellipe(np.clip(x, -0.99, 0.99)),
        "E(x)", "E(x)", complexity=5
    )

    # --- Zeta-like (for regularization in QFT) ---
    # Hurwitz zeta at s=2: ζ(2,x) = Σ 1/(n+x)²
    register_symbolic(
        "zeta_reg", _np_to_mlx(lambda x: special.zeta(2, np.abs(x) + 1)),
        lambda x: special.zeta(2, np.abs(x) + 1),
        "\\zeta(2, x)", "zeta(2, x)", complexity=5
    )


def list_physics_symbolic():
    """List all physics symbolic functions available after registration."""
    physics_functions = {
        "Quantum Mechanics": [
            "H_0", "H_1", "H_2", "H_3", "H_4",  # Hermite
            "psi_0", "psi_1", "psi_2",  # HO wavefunctions
            "L_0", "L_1", "L_2",  # Laguerre
            "R_10", "R_20", "R_21",  # Hydrogen radial
            "P_0", "P_1", "P_2", "P_3", "P_4",  # Legendre
            "T_0", "T_1", "T_2", "T_3",  # Chebyshev
            "Y_00", "cos_theta", "sin_theta",  # Spherical harmonics
        ],
        "Special Functions": [
            "J_0", "J_1", "Y_0", "Y_1",  # Bessel
            "I_0", "I_1", "K_0", "K_1",  # Modified Bessel
            "j_0", "j_1",  # Spherical Bessel
            "erf", "erfc",  # Error functions
            "gamma", "loggamma", "digamma",  # Gamma
            "Ai", "Bi",  # Airy
            "ellipK", "ellipE",  # Elliptic
        ],
        "QFT": [
            "propagator", "yukawa", "coulomb",
            "bose", "fermi", "planck",
            "Li_2", "zeta_reg",
            "lorentzian",
        ],
        "Cosmology": [
            "a_matter", "a_rad", "a_deSitter",
            "D_L", "schwarzschild",
        ],
        "Deformation Quantization": [
            "q_exp_0", "q_exp_2", "q_log_2",
            "sin_q", "cos_q",
            "moyal_1", "moyal_2",
        ],
        "General Physics": [
            "heaviside", "smooth_step", "sinc",
            "damped_sin", "damped_cos", "wave_packet",
            "x^1.5", "x^2.5", "x^-0.5", "x^-1.5",
        ],
    }
    return physics_functions
