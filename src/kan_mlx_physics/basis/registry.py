"""Basis function registry for easy instantiation.

Provides a unified interface to create any registered basis type
by name, with automatic parameter handling.

Usage:
    from kan_mlx_physics.basis import make_basis, list_bases

    # Create by name
    basis = make_basis("fourier", M=11, learnable_freq=True)

    # List available bases
    print(list_bases())  # ['bspline', 'fourier', 'chebyshev', ...]

    # Get info about a basis
    info = get_basis_info("hermite")
"""

from typing import Dict, Type, List, Any, Optional, Union
import mlx.core as mx

from .base import Basis, BasisConfig, ParamMode

# Import all basis implementations
from .bspline import BSplineBasis
from .fourier import FourierBasis, ComplexFourierBasis
from .chebyshev import ChebyshevBasis, ChebyshevSecondKindBasis
from .hermite import HermiteBasis, WeightedHermiteBasis, ProbabilistHermiteBasis
from .laguerre import LaguerreBasis, AssociatedLaguerreBasis, HydrogenRadialBasis
from .legendre import LegendreBasis, AssociatedLegendreBasis, SphericalHarmonicBasis


# Registry mapping names to classes
BASIS_REGISTRY: Dict[str, Type[Basis]] = {
    # Core bases
    "bspline": BSplineBasis,
    "fourier": FourierBasis,
    "chebyshev": ChebyshevBasis,
    "hermite": HermiteBasis,
    "laguerre": LaguerreBasis,
    "legendre": LegendreBasis,

    # Variants
    "complex_fourier": ComplexFourierBasis,
    "chebyshev_U": ChebyshevSecondKindBasis,
    "weighted_hermite": WeightedHermiteBasis,
    "probabilist_hermite": ProbabilistHermiteBasis,
    "associated_laguerre": AssociatedLaguerreBasis,
    "hydrogen_radial": HydrogenRadialBasis,
    "associated_legendre": AssociatedLegendreBasis,
    "spherical_harmonic": SphericalHarmonicBasis,
}


# Physics use case recommendations
PHYSICS_RECOMMENDATIONS: Dict[str, List[str]] = {
    "quantum_mechanics": ["hermite", "weighted_hermite", "laguerre", "hydrogen_radial"],
    "spectral_methods": ["chebyshev", "legendre", "fourier"],
    "periodic": ["fourier", "complex_fourier"],
    "bounded_interval": ["chebyshev", "legendre", "bspline"],
    "semi_infinite": ["laguerre", "hydrogen_radial"],
    "angular": ["legendre", "associated_legendre", "spherical_harmonic"],
    "oscillatory": ["fourier", "chebyshev"],
    "general": ["bspline", "chebyshev"],
}


def make_basis(
    name: str,
    M: int,
    learnable_affine: bool = True,
    param_mode: ParamMode = "per_in",
    **kwargs,
) -> Basis:
    """Create a basis by name.

    Args:
        name: Basis type name (e.g., "fourier", "chebyshev")
        M: Number of basis functions
        learnable_affine: Whether shift/scale are learnable
        param_mode: "per_in" or "per_edge" for parameter sharing
        **kwargs: Additional basis-specific parameters

    Returns:
        Initialized Basis instance

    Raises:
        ValueError: If name is not in registry

    Examples:
        >>> basis = make_basis("fourier", M=11, learnable_freq=True)
        >>> basis = make_basis("hermite", M=8, weighted=True)
        >>> basis = make_basis("bspline", M=8, k=3, grid_range=(-2, 2))
    """
    if name not in BASIS_REGISTRY:
        available = ", ".join(sorted(BASIS_REGISTRY.keys()))
        raise ValueError(f"Unknown basis: '{name}'. Available: {available}")

    # Create base config
    config = BasisConfig(
        M=M,
        learnable_affine=learnable_affine,
        param_mode=param_mode,
    )

    # Special handling for B-splines (backwards compatibility)
    if name == "bspline":
        k = kwargs.pop("k", 3)
        grid_range = kwargs.pop("grid_range", (-1.0, 1.0))
        return BSplineBasis(config, k=k, grid_range=grid_range)

    # Create basis with remaining kwargs
    basis_cls = BASIS_REGISTRY[name]
    return basis_cls(config, **kwargs)


def list_bases() -> List[str]:
    """List all available basis types.

    Returns:
        Sorted list of basis names
    """
    return sorted(BASIS_REGISTRY.keys())


def get_basis_info(name: str) -> Dict[str, Any]:
    """Get information about a basis type.

    Args:
        name: Basis name

    Returns:
        Dictionary with:
            - name: Basis name
            - class: Basis class
            - docstring: Class docstring
            - symbolic_priority: Recommended symbolic functions
            - physics_uses: Recommended physics applications
    """
    if name not in BASIS_REGISTRY:
        raise ValueError(f"Unknown basis: {name}")

    cls = BASIS_REGISTRY[name]

    # Find physics uses
    physics_uses = []
    for use, bases in PHYSICS_RECOMMENDATIONS.items():
        if name in bases:
            physics_uses.append(use)

    # Get symbolic priority from a dummy instance
    try:
        dummy_config = BasisConfig(M=5)
        dummy = cls(dummy_config)
        symbolic_priority = dummy.symbolic_priority
    except Exception:
        symbolic_priority = []

    return {
        "name": name,
        "class": cls,
        "docstring": cls.__doc__,
        "symbolic_priority": symbolic_priority,
        "physics_uses": physics_uses,
    }


def recommend_basis(
    physics_domain: str,
    domain_type: str = "bounded",
) -> List[str]:
    """Recommend bases for a physics problem.

    Args:
        physics_domain: Type of physics problem
            ("quantum_mechanics", "spectral_methods", "periodic", etc.)
        domain_type: Spatial domain type
            ("bounded", "semi_infinite", "infinite", "periodic")

    Returns:
        List of recommended basis names, in order of relevance

    Examples:
        >>> recommend_basis("quantum_mechanics", "bounded")
        ['hermite', 'weighted_hermite', 'chebyshev']
        >>> recommend_basis("periodic")
        ['fourier', 'complex_fourier']
    """
    recommendations = []

    # Get domain-specific recommendations
    if physics_domain in PHYSICS_RECOMMENDATIONS:
        recommendations.extend(PHYSICS_RECOMMENDATIONS[physics_domain])

    # Add domain-type recommendations
    domain_map = {
        "bounded": ["chebyshev", "legendre", "bspline"],
        "semi_infinite": ["laguerre", "hydrogen_radial"],
        "infinite": ["hermite", "fourier"],
        "periodic": ["fourier", "complex_fourier"],
    }

    if domain_type in domain_map:
        for basis in domain_map[domain_type]:
            if basis not in recommendations:
                recommendations.append(basis)

    # Ensure uniqueness while preserving order
    seen = set()
    unique = []
    for r in recommendations:
        if r not in seen:
            seen.add(r)
            unique.append(r)

    return unique if unique else ["bspline"]  # Default fallback


def register_basis(name: str, cls: Type[Basis]) -> None:
    """Register a custom basis type.

    Args:
        name: Name to register under
        cls: Basis subclass

    Raises:
        TypeError: If cls is not a Basis subclass
        ValueError: If name already registered
    """
    if not issubclass(cls, Basis):
        raise TypeError(f"{cls} is not a Basis subclass")

    if name in BASIS_REGISTRY:
        raise ValueError(f"Basis '{name}' already registered")

    BASIS_REGISTRY[name] = cls
