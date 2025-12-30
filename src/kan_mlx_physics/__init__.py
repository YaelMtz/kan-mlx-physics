"""KAN-MLX-Physics: Physics-First Kolmogorov-Arnold Networks for Apple Silicon.

High-performance implementation of Kolmogorov-Arnold Networks optimized for Apple Silicon.
Provides specialized tools for physics-informed neural networks (PINNs) and PDE solving.

Quick start (convenience API):
    from kan_mlx_physics import quick_fit, auto_formula, from_preset

    # One-liner training
    model, history = quick_fit(
        f=lambda x: mx.sin(mx.pi * x[:, 0]) + x[:, 1]**2,
        n_var=2,
    )
    formula = auto_formula(model, history['x_sample'], ['x', 'y'])

    # Physics presets
    model = from_preset("quantum_oscillator", width=[1, 10, 1])
    model = from_preset("wave_equation", width=[2, 15, 1])

Standard API:
    from kan_mlx_physics import MultKAN, create_dataset

    model = MultKAN(width=[2, 5, 1])
    model = MultKAN(width=[2, 5, 1], basis="fourier", basis_M=11)
    model = MultKAN(width=[1, 10, 1], basis="hermite", basis_kwargs={"weighted": True})

Supports pluggable basis functions:
    - B-splines (default, general purpose)
    - Fourier (periodic/oscillatory, wave equations)
    - Chebyshev (spectral methods, bounded intervals)
    - Hermite (quantum harmonic oscillator)
    - Laguerre (radial problems, hydrogen atom)
    - Legendre (angular momentum, spherical harmonics)
"""

# =============================================================================
# Core Model
# =============================================================================
from .multkan import MultKAN, TrainingDataset, TrainingHistory
from .kan_layer import KANLayer

# =============================================================================
# Convenience API (High-Level)
# =============================================================================
from .convenience import quick_fit, auto_formula, visualize, fit_and_extract
from .presets import from_preset, list_presets, describe_preset, get_preset_config
from .utils import (
    create_dataset,
    create_dataset_from_data,
    f_sin,
    f_exp,
    f_gaussian,
    f_polynomial,
    f_mixed,
)

# =============================================================================
# Basis Functions
# =============================================================================
from . import basis
from .basis import (
    make_basis,
    list_bases,
    recommend_basis,
    Basis,
    BasisConfig,
)

# =============================================================================
# Symbolic Regression
# =============================================================================
from .symbolic import (
    Symbolic_KANLayer,
    SymbolicFunction,
    add_symbolic,
    list_symbolic,
    fit_affine_params,
    suggest_symbolic,
    validate_domain,
    score_symbolic_fit,
    SYMBOLIC_REGISTRY,
)
from .physics_symbolic import (
    register_physics_symbolic,
    list_physics_symbolic,
)

# =============================================================================
# Visualization
# =============================================================================
from .visualization import (
    plot,
    plot_kan,
    plot_activations,
    plot_training_history,
    plot_spline_1d,
)

# =============================================================================
# Functional API (Advanced)
# =============================================================================
from .functional import (
    LayerParams,
    ParamsList,
    kan_layer_forward,
    functional_forward,
    get_params_list,
    set_params_list,
    init_adam_state,
    adam_update,
    flatten_params,
    unflatten_params,
    lbfgs_optimize,
    lbfgs_fit,
)

# =============================================================================
# PINN Utilities (Physics-Informed Neural Networks)
# =============================================================================
from .pinn import (
    make_u_fn,
    make_derivative_fns,
    make_laplacian_fn,
    make_gradient_fn,
    finite_difference_laplacian,
    PINNOperators,
    make_compiled_pinn_step,
)

# =============================================================================
# Formula Rendering
# =============================================================================
from .formula_render import (
    FormulaRenderer,
    FormulaTerm,
    render_formula,
    print_formula_box,
)

# =============================================================================
# Low-Level Spline Functions
# =============================================================================
from .spline import B_batch, coef2curve, curve2coef, extend_grid

# =============================================================================
# Submodules
# =============================================================================
from . import pde
from . import functional
from . import pinn

# =============================================================================
# Version
# =============================================================================
__version__ = "0.1.0"
__all__ = [
    # =========================================================================
    # Core Model
    # =========================================================================
    "MultKAN",
    "KANLayer",
    "TrainingDataset",
    "TrainingHistory",

    # =========================================================================
    # Convenience API (High-Level) - Start here!
    # =========================================================================
    "quick_fit",
    "auto_formula",
    "visualize",
    "fit_and_extract",
    # Presets
    "from_preset",
    "list_presets",
    "describe_preset",
    "get_preset_config",
    # Dataset utilities
    "create_dataset",
    "create_dataset_from_data",
    "f_sin",
    "f_exp",
    "f_gaussian",
    "f_polynomial",
    "f_mixed",

    # =========================================================================
    # Basis Functions
    # =========================================================================
    "basis",
    "make_basis",
    "list_bases",
    "recommend_basis",
    "Basis",
    "BasisConfig",

    # =========================================================================
    # Symbolic Regression
    # =========================================================================
    "Symbolic_KANLayer",
    "SymbolicFunction",
    "add_symbolic",
    "list_symbolic",
    "fit_affine_params",
    "suggest_symbolic",
    "validate_domain",
    "score_symbolic_fit",
    "SYMBOLIC_REGISTRY",
    # Physics symbolic
    "register_physics_symbolic",
    "list_physics_symbolic",

    # =========================================================================
    # Visualization
    # =========================================================================
    "plot",
    "plot_kan",
    "plot_activations",
    "plot_training_history",
    "plot_spline_1d",

    # =========================================================================
    # Functional API (Advanced)
    # =========================================================================
    "functional",
    "LayerParams",
    "ParamsList",
    "kan_layer_forward",
    "functional_forward",
    "get_params_list",
    "set_params_list",
    "init_adam_state",
    "adam_update",
    "flatten_params",
    "unflatten_params",
    "lbfgs_optimize",
    "lbfgs_fit",

    # =========================================================================
    # PINN Utilities
    # =========================================================================
    "pinn",
    "make_u_fn",
    "make_derivative_fns",
    "make_laplacian_fn",
    "make_gradient_fn",
    "finite_difference_laplacian",
    "PINNOperators",
    "make_compiled_pinn_step",

    # =========================================================================
    # Formula Rendering
    # =========================================================================
    "FormulaRenderer",
    "FormulaTerm",
    "render_formula",
    "print_formula_box",

    # =========================================================================
    # PDE Module
    # =========================================================================
    "pde",

    # =========================================================================
    # Low-Level Spline Functions
    # =========================================================================
    "B_batch",
    "coef2curve",
    "curve2coef",
    "extend_grid",
]
