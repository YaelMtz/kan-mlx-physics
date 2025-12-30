"""Pluggable basis functions for KAN edge parameterization.

This module provides a modular system for basis functions in
Kolmogorov-Arnold Networks. Each basis type is suited for different
physics problems:

    - B-splines: General-purpose, adaptive grid
    - Fourier: Periodic/oscillatory, wave equations
    - Chebyshev: Spectral methods, bounded intervals
    - Hermite: Quantum harmonic oscillator, Gaussian-weighted
    - Laguerre: Radial problems, hydrogen atom
    - Legendre: Angular momentum, spherical harmonics

Quick start:
    from kan_mlx_physics.basis import make_basis, list_bases

    # Create a Fourier basis with 11 functions
    basis = make_basis("fourier", M=11, learnable_freq=True)

    # Initialize parameters
    params = basis.init_params(in_dim=5)

    # Evaluate basis functions
    x = mx.random.uniform(shape=(32, 5))
    phis = basis.features(x, params)  # shape: (32, 5, 11)

For physics applications:
    from kan_mlx_physics.basis import recommend_basis

    # Get recommendations for quantum mechanics
    bases = recommend_basis("quantum_mechanics", "bounded")
    # ['hermite', 'weighted_hermite', 'chebyshev']
"""

# Base classes and utilities
from .base import (
    Basis,
    BasisConfig,
    ParamMode,
    softplus,
    contract_basis_coef,
)

# Specific basis implementations
from .bspline import BSplineBasis
from .fourier import FourierBasis, ComplexFourierBasis
from .chebyshev import ChebyshevBasis, ChebyshevSecondKindBasis
from .hermite import HermiteBasis, WeightedHermiteBasis, ProbabilistHermiteBasis
from .laguerre import LaguerreBasis, AssociatedLaguerreBasis, HydrogenRadialBasis
from .legendre import LegendreBasis, AssociatedLegendreBasis, SphericalHarmonicBasis

# Registry functions
from .registry import (
    make_basis,
    list_bases,
    get_basis_info,
    recommend_basis,
    register_basis,
    BASIS_REGISTRY,
    PHYSICS_RECOMMENDATIONS,
)


__all__ = [
    # Base
    "Basis",
    "BasisConfig",
    "ParamMode",
    "softplus",
    "contract_basis_coef",

    # B-spline
    "BSplineBasis",

    # Fourier
    "FourierBasis",
    "ComplexFourierBasis",

    # Chebyshev
    "ChebyshevBasis",
    "ChebyshevSecondKindBasis",

    # Hermite
    "HermiteBasis",
    "WeightedHermiteBasis",
    "ProbabilistHermiteBasis",

    # Laguerre
    "LaguerreBasis",
    "AssociatedLaguerreBasis",
    "HydrogenRadialBasis",

    # Legendre
    "LegendreBasis",
    "AssociatedLegendreBasis",
    "SphericalHarmonicBasis",

    # Registry
    "make_basis",
    "list_bases",
    "get_basis_info",
    "recommend_basis",
    "register_basis",
    "BASIS_REGISTRY",
    "PHYSICS_RECOMMENDATIONS",
]
