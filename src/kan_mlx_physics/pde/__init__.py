"""PDE solving module for KAN-MLX-Physics.

Physics-first approach to solving PDEs with Kolmogorov-Arnold Networks.
Designed for cosmological models and deformation quantization.

Core philosophy:
    Physics first, mathematics second, code third.

Quick start:
    from kan_mlx_physics.pde import solve

    # Type your equation, get a solution
    ψ, history = solve("−∇²ψ/2 + x²ψ/2 = Eψ", domain=[-5, 5], params={"E": 0.5})

    # Wheeler-DeWitt with Moyal deformation
    Ψ, _ = solve("Ĥ ⋆_θ Ψ = 0", domain=[0.1, 5], params={"θ": 0.05})

    # Use templates
    φ, _ = solve("klein-gordon", domain={"t": [0, 5], "x": [-5, 5]})
"""

from .operators import (
    # Differential operators
    grad,
    laplacian,
    div,
    curl,
    # Star products
    moyal_bracket,
    star_product,
    # Cosmological operators
    friedmann_operator,
    wheeler_dewitt_operator,
    klein_gordon_curved,
)

from .problem import (
    PDEProblem,
    BoundaryCondition,
    Domain,
)

from .physics import (
    # Cosmology
    WheelerDeWitt,
    Friedmann,
    KleinGordonCurved,
    FriedmannPerturbations,
    MultiFieldWheelerDeWitt,
    # Quantum mechanics
    Schrodinger,
    DeformedSchrodinger,
    # Deformed cosmology
    DeformedWheelerDeWitt,
)

from .solver import solve as solve_problem, SolverConfig, adaptive_solve

from .complex import (
    ComplexWavefunction,
    complex_residual,
    complex_bc_loss,
    wkb_phase_loss,
)

from .symbolic_extraction import (
    ExtractedFormula,
    WKBExtraction,
    extract_symbolic_solution,
    validate_symbolic_solution,
    symbolic_wkb_extraction,
    get_physics_symbolic_library,
)

from .dsl import (
    solve,
    equation,
    list_equations,
    help_equation,
    # Shortcuts
    schrodinger,
    wheeler_dewitt,
    klein_gordon,
    wave,
)

__all__ = [
    # DSL - Primary interface
    "solve",
    "equation",
    "list_equations",
    "help_equation",
    # Shortcuts
    "schrodinger",
    "wheeler_dewitt",
    "klein_gordon",
    "wave",
    # Operators
    "grad",
    "laplacian",
    "div",
    "curl",
    "moyal_bracket",
    "star_product",
    "friedmann_operator",
    "wheeler_dewitt_operator",
    "klein_gordon_curved",
    # Problem definition
    "PDEProblem",
    "BoundaryCondition",
    "Domain",
    # Physics problems
    "WheelerDeWitt",
    "Friedmann",
    "KleinGordonCurved",
    "FriedmannPerturbations",
    "MultiFieldWheelerDeWitt",
    "Schrodinger",
    "DeformedSchrodinger",
    "DeformedWheelerDeWitt",
    # Solver
    "solve_problem",
    "SolverConfig",
    "adaptive_solve",
    # Complex wavefunctions
    "ComplexWavefunction",
    "complex_residual",
    "complex_bc_loss",
    "wkb_phase_loss",
    # Symbolic extraction
    "ExtractedFormula",
    "WKBExtraction",
    "extract_symbolic_solution",
    "validate_symbolic_solution",
    "symbolic_wkb_extraction",
    "get_physics_symbolic_library",
]
