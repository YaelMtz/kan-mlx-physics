"""PDE solving module for KAN-MLX-Physics.

Physics-first approach to solving PDEs with Kolmogorov-Arnold Networks.
Designed for cosmological models and deformation quantization.

Core philosophy:
    Physics first, mathematics second, code third.

Quick start (new DSL - recommended):
    from kan_mlx_physics.pde import pde, PDEBuilder

    # One-liner: type your equation, get a solution
    model, history = pde("−∇²ψ/2 + x²ψ/2 = Eψ", domain=[-5, 5], params={"E": 0.5})

    # Builder API for full control
    model, history = (
        PDEBuilder("-∇²ψ/2 + V(x)ψ = Eψ")
        .params(E=0.5, V=lambda x: 0.5 * x**2)
        .domain([-5, 5])
        .bc("dirichlet")
        .loss(NormalizationLoss(weight=10))
        .loss(NonTrivialLoss(weight=20))
        .phase("warmup", steps=1000, lr=0.01)
        .phase("refine", steps=500, lr=0.001, grid_update_before=True)
        .model(width=[1, 20, 20, 1], basis="hermite")
        .solve()
    )

    # Use shortcuts for common equations
    model, _ = schrodinger(domain=[-5, 5], params={"E": 0.5})
    model, _ = wheeler_dewitt(domain=[0.1, 5], params={"Lambda": 0.1})

Multi-phase training example:
    from kan_mlx_physics.pde import (
        pde, PDEBuilder, TrainingSchedule, TrainingPhase,
        PDEResidualLoss, NormalizationLoss, EigenvalueLoss,
    )

    # Define custom training schedule
    schedule = TrainingSchedule([
        TrainingPhase("warmup", steps=1000, lr=0.01),
        TrainingPhase("refine", steps=500, lr=0.001, grid_update_before=True),
        TrainingPhase("fine-tune", steps=200, lr=0.0001, symbolic_after=True),
    ])

    model, history = pde(
        "∇²u = f",
        domain=[-1, 1],
        schedule=schedule,
    )
"""

# =============================================================================
# New DSL (Primary Interface) - Recommended
# =============================================================================
from .dsl import (
    # Main entry point
    pde,
    # Builder API
    PDEBuilder,
    # Shortcuts
    schrodinger,
    wheeler_dewitt,
    klein_gordon,
    wave,
    heat,
    laplace,
    poisson,
    # Utility functions
    list_equations,
    help_equation,
    list_presets,
    describe_preset,
)

# =============================================================================
# Expression IR and Parser
# =============================================================================
from .expr import (
    PDEExpr,
    Variable,
    Coordinate,
    Parameter,
    Constant,
    Derivative,
    Laplacian,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    ParsedPDE,
)

from .parser import (
    PDEParser,
    FormatDetector,
    TEMPLATES,
)

# =============================================================================
# Analysis
# =============================================================================
from .analysis import (
    DerivativeSpec,
    JetRequirements,
    JetExtractor,
    EquationType,
    EquationClassifier,
    PDEAnalysis,
)

# =============================================================================
# Compiler
# =============================================================================
from .compiler import (
    JetCache,
    CompiledResidual,
    ResidualCompiler,
)

# =============================================================================
# Loss System
# =============================================================================
from .spectrum import solve_spectrum, Spectrum, DeflationLoss
from .priors import Prior, compare_priors, PriorComparison
from .losses import (
    TraceLoss,
    PurityLoss,
    # Context
    LossContext,
    # Base class
    LossTerm,
    # Composer
    LossComposer,
    # Built-in losses
    PDEResidualLoss,
    BoundaryConditionLoss,
    NormalizationLoss,
    NonTrivialLoss,
    EigenvalueLoss,
    SmoothnessLoss,
    DecayLoss,
    AnchorLoss,
    RegularizationLoss,
    DataLoss,
)

# =============================================================================
# Multi-Phase Trainer
# =============================================================================
from .trainer import (
    TrainingPhase,
    TrainingSchedule,
    PhaseHistory,
    TrainingHistory,
    PDETrainer,
    GradNormBalancer,
)

# =============================================================================
# Hard Constraints (Ansatz)
# =============================================================================
from .constraints import (
    HardConstraint,
    SineSeriesConstraint,
    wrap_model_with_constraint,
    wrap_functional_with_constraint,
)

# =============================================================================
# Operators (Existing)
# =============================================================================
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

# =============================================================================
# Problem Definition (Existing)
# =============================================================================
from .problem import (
    PDEProblem,
    BoundaryCondition,
    Domain,
)

# =============================================================================
# Physics Problems (Existing)
# =============================================================================
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

# =============================================================================
# Solver (Existing)
# =============================================================================
from .solver import solve as solve_problem, SolverConfig, adaptive_solve

# =============================================================================
# Complex Wavefunctions (Existing)
# =============================================================================
from .complex import (
    ComplexWavefunction,
    complex_residual,
    complex_bc_loss,
    wkb_phase_loss,
)

# =============================================================================
# Symbolic Extraction (Existing)
# =============================================================================
from .symbolic_extraction import (
    ExtractedFormula,
    WKBExtraction,
    extract_symbolic_solution,
    validate_symbolic_solution,
    symbolic_wkb_extraction,
    get_physics_symbolic_library,
)

__all__ = [
    "PriorComparison",
    "compare_priors",
    "Prior",
    "PurityLoss",
    "TraceLoss",
    "DeflationLoss",
    "Spectrum",
    "solve_spectrum",
    # =========================================================================
    # New DSL - Primary Interface (Start here!)
    # =========================================================================
    "pde",
    "PDEBuilder",
    # Shortcuts
    "schrodinger",
    "wheeler_dewitt",
    "klein_gordon",
    "wave",
    "heat",
    "laplace",
    "poisson",
    # Utilities
    "list_equations",
    "help_equation",
    "list_presets",
    "describe_preset",

    # =========================================================================
    # Expression IR
    # =========================================================================
    "PDEExpr",
    "Variable",
    "Coordinate",
    "Parameter",
    "Constant",
    "Derivative",
    "Laplacian",
    "BinaryOp",
    "UnaryOp",
    "FunctionCall",
    "ParsedPDE",

    # =========================================================================
    # Parser
    # =========================================================================
    "PDEParser",
    "FormatDetector",
    "TEMPLATES",

    # =========================================================================
    # Analysis
    # =========================================================================
    "DerivativeSpec",
    "JetRequirements",
    "JetExtractor",
    "EquationType",
    "EquationClassifier",
    "PDEAnalysis",

    # =========================================================================
    # Compiler
    # =========================================================================
    "JetCache",
    "CompiledResidual",
    "ResidualCompiler",

    # =========================================================================
    # Loss System
    # =========================================================================
    "LossContext",
    "LossTerm",
    "LossComposer",
    "PDEResidualLoss",
    "BoundaryConditionLoss",
    "NormalizationLoss",
    "NonTrivialLoss",
    "EigenvalueLoss",
    "SmoothnessLoss",
    "DecayLoss",
    "AnchorLoss",
    "RegularizationLoss",
    "DataLoss",

    # =========================================================================
    # Multi-Phase Trainer
    # =========================================================================
    "TrainingPhase",
    "TrainingSchedule",
    "PhaseHistory",
    "TrainingHistory",
    "PDETrainer",
    "GradNormBalancer",

    # =========================================================================
    # Hard Constraints (Ansatz)
    # =========================================================================
    "HardConstraint",
    "SineSeriesConstraint",
    "wrap_model_with_constraint",
    "wrap_functional_with_constraint",

    # =========================================================================
    # Operators
    # =========================================================================
    "grad",
    "laplacian",
    "div",
    "curl",
    "moyal_bracket",
    "star_product",
    "friedmann_operator",
    "wheeler_dewitt_operator",
    "klein_gordon_curved",

    # =========================================================================
    # Problem Definition
    # =========================================================================
    "PDEProblem",
    "BoundaryCondition",
    "Domain",

    # =========================================================================
    # Physics Problems
    # =========================================================================
    "WheelerDeWitt",
    "Friedmann",
    "KleinGordonCurved",
    "FriedmannPerturbations",
    "MultiFieldWheelerDeWitt",
    "Schrodinger",
    "DeformedSchrodinger",
    "DeformedWheelerDeWitt",

    # =========================================================================
    # Solver
    # =========================================================================
    "solve_problem",
    "SolverConfig",
    "adaptive_solve",

    # =========================================================================
    # Complex Wavefunctions
    # =========================================================================
    "ComplexWavefunction",
    "complex_residual",
    "complex_bc_loss",
    "wkb_phase_loss",

    # =========================================================================
    # Symbolic Extraction
    # =========================================================================
    "ExtractedFormula",
    "WKBExtraction",
    "extract_symbolic_solution",
    "validate_symbolic_solution",
    "symbolic_wkb_extraction",
    "get_physics_symbolic_library",
]
