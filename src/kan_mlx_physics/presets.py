"""Physics-aware model presets for common use cases.

Presets provide pre-configured basis functions optimized for specific
physics domains, reducing boilerplate and ensuring best practices.

Example:
    >>> from kan_mlx_physics import from_preset
    >>> model = from_preset("quantum_oscillator", width=[1, 10, 1])
    >>> model = from_preset("wave_equation", width=[2, 15, 1], basis_M=31)
"""

from typing import Dict, Any, List, Optional

# Lazy import to avoid circular dependencies
def _get_multkan():
    from .multkan import MultKAN
    return MultKAN


# Physics preset configurations
PRESETS: Dict[str, Dict[str, Any]] = {
    "quantum_oscillator": {
        "basis": "hermite",
        "basis_kwargs": {"weighted": True},
        "basis_M": 8,
        "description": "Quantum harmonic oscillator wavefunctions",
        "domain": "Quantum Mechanics",
        "use_case": "Schrodinger equation with quadratic potential",
    },
    "wave_equation": {
        "basis": "fourier",
        "basis_M": 21,
        "basis_kwargs": {"learnable_freq": True},
        "description": "Periodic/oscillatory solutions (waves, vibrations)",
        "domain": "Wave Physics",
        "use_case": "Wave equations, Helmholtz, standing waves",
    },
    "spectral": {
        "basis": "chebyshev",
        "basis_M": 12,
        "description": "High-accuracy spectral methods on bounded domains",
        "domain": "Numerical Methods",
        "use_case": "Spectral collocation, boundary value problems",
    },
    "radial": {
        "basis": "laguerre",
        "basis_kwargs": {"alpha": 0.0},
        "basis_M": 10,
        "description": "Radial problems (hydrogen atom, spherical coords)",
        "domain": "Quantum Mechanics",
        "use_case": "Hydrogen wavefunctions, radial Schrodinger",
    },
    "angular": {
        "basis": "legendre",
        "basis_M": 8,
        "description": "Angular momentum, spherical harmonics",
        "domain": "Quantum Mechanics",
        "use_case": "Spherical harmonics, Legendre ODEs",
    },
    "general": {
        "basis": "bspline",
        "grid": 5,
        "k": 3,
        "description": "General-purpose adaptive B-spline (default)",
        "domain": "General",
        "use_case": "Function approximation, symbolic regression",
    },
}


def from_preset(
    preset: str,
    width: List[int],
    **overrides
) -> "MultKAN":
    """Create a MultKAN model from a physics preset.

    Presets provide optimized configurations for common physics problems:
    - quantum_oscillator: Hermite basis for harmonic oscillator wavefunctions
    - wave_equation: Fourier basis for periodic/oscillatory solutions
    - spectral: Chebyshev basis for high-accuracy spectral methods
    - radial: Laguerre basis for radial problems (hydrogen atom)
    - angular: Legendre basis for angular momentum problems
    - general: B-spline basis for general function approximation

    Args:
        preset: Name of the preset to use
        width: Network architecture [input_dim, hidden..., output_dim]
        **overrides: Override any preset parameter (e.g., basis_M=15)

    Returns:
        Configured MultKAN model

    Example:
        >>> model = from_preset("quantum_oscillator", width=[1, 10, 1])
        >>> model = from_preset("wave_equation", width=[2, 15, 1], basis_M=31)
        >>> model = from_preset("spectral", width=[2, 8, 8, 1])
    """
    if preset not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    # Get preset config and remove metadata
    config = {**PRESETS[preset]}
    config.pop("description", None)
    config.pop("domain", None)
    config.pop("use_case", None)

    # Apply user overrides
    config.update(overrides)

    MultKAN = _get_multkan()
    return MultKAN(width=width, **config)


def list_presets() -> Dict[str, str]:
    """List available presets with their descriptions.

    Returns:
        Dictionary mapping preset names to descriptions

    Example:
        >>> presets = list_presets()
        >>> for name, desc in presets.items():
        ...     print(f"{name}: {desc}")
    """
    return {name: cfg["description"] for name, cfg in PRESETS.items()}


def describe_preset(preset: str) -> str:
    """Get detailed description of a preset configuration.

    Args:
        preset: Name of the preset

    Returns:
        Multi-line description including all configuration details

    Example:
        >>> print(describe_preset("quantum_oscillator"))
        Preset: quantum_oscillator
        Description: Quantum harmonic oscillator wavefunctions
        ...
    """
    if preset not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    cfg = PRESETS[preset]
    lines = [
        f"Preset: {preset}",
        f"Description: {cfg['description']}",
        f"Domain: {cfg.get('domain', 'General')}",
        f"Use case: {cfg.get('use_case', 'N/A')}",
        "",
        "Configuration:",
        f"  Basis: {cfg.get('basis', 'bspline')}",
    ]

    if "basis_M" in cfg:
        lines.append(f"  Basis functions (M): {cfg['basis_M']}")
    if "basis_kwargs" in cfg:
        lines.append(f"  Basis options: {cfg['basis_kwargs']}")
    if "grid" in cfg:
        lines.append(f"  Grid points: {cfg['grid']}")
    if "k" in cfg:
        lines.append(f"  Spline order: {cfg['k']}")

    return "\n".join(lines)


def get_preset_config(preset: str) -> Dict[str, Any]:
    """Get the raw configuration dictionary for a preset.

    Args:
        preset: Name of the preset

    Returns:
        Configuration dictionary (excluding metadata)

    Example:
        >>> config = get_preset_config("wave_equation")
        >>> config['basis']
        'fourier'
    """
    if preset not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{preset}'. Available: {available}")

    config = {**PRESETS[preset]}
    # Remove metadata fields
    config.pop("description", None)
    config.pop("domain", None)
    config.pop("use_case", None)

    return config
