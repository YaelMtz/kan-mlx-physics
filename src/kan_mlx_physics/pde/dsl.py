"""Physics DSL: Type PDEs naturally, solve them with KANs.

New general-purpose DSL that supports:
- Arbitrary PDE parsing with SymPy backend
- Auto-detection of unknowns, parameters, and derivatives
- Modular loss composition
- Multi-phase training
- Full control via PDEBuilder API

Examples:
    # Simple usage
    u, history = pde("∇²u = 0", domain=[-1, 1])

    # With parameters
    ψ, history = pde("-∇²ψ/2 + x²ψ/2 = Eψ", domain=[-5, 5], params={"E": 0.5})

    # Full control with PDEBuilder
    model, history = (
        PDEBuilder("-∇²ψ/2 + x²ψ/2 = Eψ")
        .params(E=0.5, hbar=1.0)
        .domain([-5, 5])
        .loss(NormalizationLoss(weight=10))
        .phase("warmup", steps=1000, lr=0.01)
        .phase("refine", steps=500, lr=0.001)
        .model(width=[1, 20, 20, 1], basis="hermite")
        .solve()
    )

Physics first. Always.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable, Union, Any, Set, Literal
from dataclasses import dataclass, field
from pathlib import Path

# Import new components
from .expr import PDEExpr, ParsedPDE
from .parser import PDEParser, parse, get_template, list_templates, TEMPLATES
from .analysis import PDEAnalysis, JetRequirements, analyze, JetExtractor, EquationClassifier
from .compiler import CompiledResidual, ResidualCompiler, compile_residual
from .losses import (
    LossTerm, LossComposer, LossContext,
    PDEResidualLoss, BoundaryConditionLoss, NormalizationLoss,
    NonTrivialLoss, EigenvalueLoss, SmoothnessLoss, DecayLoss,
    AnchorLoss, RegularizationLoss, DataLoss,
    default_pinn_losses, default_eigenvalue_losses, default_quantum_losses,
)
from .trainer import (
    PDETrainer, TrainingPhase, TrainingSchedule, TrainingHistory,
    create_trainer, train_pde,
)
from .problem import Domain, BoundaryCondition


# =============================================================================
# MAIN DSL FUNCTION
# =============================================================================

def pde(
    equation: str,
    domain: Optional[Union[List, Tuple, Dict, Domain]] = None,
    params: Optional[Dict[str, float]] = None,
    trainable_param_names: Optional[Set[str]] = None,
    bc: Optional[Union[str, List[BoundaryCondition]]] = None,
    schedule: Optional[TrainingSchedule] = None,
    losses: Optional[List[LossTerm]] = None,
    model_config: Optional[Dict] = None,
    steps: Optional[int] = None,
    verbose: bool = True,
    format: str = "auto",
) -> Tuple[Any, TrainingHistory]:
    """Solve a PDE from natural notation with full control.

    This is the main entry point for the PDE DSL. It parses the equation,
    analyzes it, compiles the residual, and trains a KAN model.

    Args:
        equation: Equation string in any supported format (Unicode, LaTeX,
                 Typst, subscript, SymPy) or a template name (e.g., "schrodinger").
        format: Format hint for parsing ('auto', 'unicode', 'latex', 'typst',
                'subscript', 'sympy', 'natural'). Default: 'auto'.
        domain: Domain specification. Can be:
                - [a, b] for 1D interval
                - [(a1, b1), (a2, b2), ...] for multi-D box
                - {"x": [a, b], "t": [c, d]} for named coordinates
                - Domain object
        params: Parameter values (E, hbar, m, theta, etc.).
        bc: Boundary conditions. Can be:
            - "dirichlet", "neumann", "periodic"
            - List of BoundaryCondition objects
        schedule: Training schedule (TrainingSchedule object).
        losses: List of loss terms to use.
        model_config: Model configuration dict with keys like:
                     - width: [input_dim, hidden, ..., output_dim]
                     - basis: "bspline", "hermite", "laguerre", etc.
                     - grid: Number of grid points
                     - k: Spline order
        steps: Total training steps (shortcut for simple schedules).
        verbose: Whether to print progress.

    Returns:
        Tuple of (trained_model, training_history).

    Examples:
        # Simple usage
        >>> u, history = pde("∇²u = 0", domain=[-1, 1])

        # Schrödinger equation
        >>> psi, history = pde(
        ...     "-∇²ψ/2 + x²ψ/2 = Eψ",
        ...     domain=[-5, 5],
        ...     params={"E": 0.5}
        ... )

        # Using template
        >>> psi, history = pde("schrodinger", domain=[-5, 5])

        # With custom losses
        >>> from kan_mlx_physics.pde import NormalizationLoss, NonTrivialLoss
        >>> psi, history = pde(
        ...     "-∇²ψ/2 + x²ψ/2 = Eψ",
        ...     domain=[-5, 5],
        ...     losses=[
        ...         PDEResidualLoss(weight=1.0),
        ...         NormalizationLoss(weight=10.0),
        ...         NonTrivialLoss(weight=5.0),
        ...     ]
        ... )

        # With custom schedule
        >>> schedule = TrainingSchedule.eigenvalue(total_steps=3000)
        >>> psi, history = pde("schrodinger", domain=[-5, 5], schedule=schedule)
    """
    # Check if equation is a template name
    equation_lower = equation.lower().replace('-', '_').replace(' ', '_')
    if equation_lower in TEMPLATES:
        equation = TEMPLATES[equation_lower]

    # Parse equation
    parser = PDEParser()
    parsed = parser.parse(equation, format=format)

    # Analyze equation
    analysis = analyze(parsed)

    # Merge parameters
    all_params = {**analysis.parameters}
    if params:
        all_params.update(params)

    # Build domain
    domain_obj = _build_domain(domain, analysis.coordinates)

    # Compile residual with derivative method configuration
    derivative_method = "autodiff"
    finite_diff_h = 1e-4
    if model_config:
        derivative_method = model_config.get('_derivative_method', 'autodiff')
        finite_diff_h = model_config.get('_finite_diff_h', 1e-4)

    compiled = compile_residual(
        parsed, analysis.jet_req,
        derivative_method=derivative_method,
        finite_diff_h=finite_diff_h
    )

    # Register custom functions if provided via model_config
    if model_config and '_custom_functions' in model_config:
        for name, fn in model_config['_custom_functions'].items():
            compiled.register_function(name, fn)

    # Build loss composer
    composer = _build_loss_composer(losses, analysis)

    # Build training schedule
    if schedule is None:
        if steps is not None:
            # Simple schedule with specified steps
            schedule = TrainingSchedule.default_pinn(total_steps=steps)
        elif analysis.is_eigenvalue_problem:
            schedule = TrainingSchedule.eigenvalue()
        else:
            schedule = TrainingSchedule.default_pinn()

    # Pop live plotter before model creation (must not leak into model config)
    live_plotter = model_config.pop("_live_plotter", None) if model_config else None
    if live_plotter is not None:
        live_plotter.pre_attach(domain_obj)

    # Create model
    model = _create_model(model_config, domain_obj, analysis)

    # Create trainer
    trainer = PDETrainer(
        model=model,
        domain=domain_obj,
        compiled_residual=compiled,
        loss_composer=composer,
        schedule=schedule,
        params=all_params,
        trainable_param_names=trainable_param_names,
    )

    # Wire live plotter into trainer
    if live_plotter is not None:
        trainer._live_plotter = live_plotter

    # Train
    history = trainer.train(verbose=verbose)

    return model, history


# =============================================================================
# PDE BUILDER (Fluent API)
# =============================================================================

class PDEBuilder:
    """Fluent builder for complex PDE problems.

    Provides a chainable API for configuring all aspects of PDE solving.

    Example:
        >>> model, history = (
        ...     PDEBuilder("-∇²ψ/2 + x²ψ/2 = Eψ")
        ...     .params(E=0.5, hbar=1.0)
        ...     .domain([-5, 5])
        ...     .bc("dirichlet")
        ...     .loss(NormalizationLoss(weight=10))
        ...     .loss(NonTrivialLoss(weight=20))
        ...     .phase("warmup", steps=1000, lr=0.01)
        ...     .phase("refine", steps=500, lr=0.001, grid_update_before=True)
        ...     .model(width=[1, 20, 20, 1], basis="hermite")
        ...     .solve()
        ... )
    """

    def __init__(self, equation: str, format: str = "auto"):
        """Initialize builder with equation string.

        Args:
            equation: PDE equation in any supported format.
            format: Format hint for parsing. Options:
                - 'auto': Auto-detect format (default)
                - 'unicode': Unicode notation (∇²ψ, ∂ψ/∂x)
                - 'latex': LaTeX notation (\\nabla^2 \\psi)
                - 'typst': Typst notation (nabla^2 psi, frac(a, b))
                - 'subscript': Subscript notation (u_xx, psi_t)
                - 'sympy': SymPy notation (Derivative(u, x, 2))
                - 'natural': Natural notation (laplacian(u))
        """
        self._equation = equation
        self._format = format
        self._params: Dict[str, float] = {}
        self._trainable_param_names: Set[str] = set()  # NEW: Track trainable params
        self._domain: Optional[Union[List, Tuple, Dict, Domain]] = None
        self._bcs: List[Any] = []
        self._losses: List[LossTerm] = []
        self._phases: List[TrainingPhase] = []
        self._model_config: Dict = {}
        self._verbose: bool = True
        self._preset_name: Optional[str] = None
        self._optimizer_config: Dict[str, Any] = {"type": "adam"}
        self._checkpoint_path: Optional[str] = None
        self._visualize_phases: List[str] = []
        self._trained_model: Optional[Any] = None  # Store for visualization
        self._custom_functions: Dict[str, Callable] = {}  # Custom functions registry
        self._derivative_method: str = "autodiff"  # Default to autodiff for accuracy
        self._finite_diff_h: float = 1e-4  # Step size for finite differences
        self._live_plotter = None  # Optional LivePlotter for live visualization

    def params(self, **kwargs) -> "PDEBuilder":
        """Set parameter values.

        Args:
            **kwargs: Parameter name-value pairs (E=0.5, hbar=1.0, etc.).

        Returns:
            Self for chaining.
        """
        self._params.update(kwargs)
        return self

    def trainable_params(self, *names: str) -> "PDEBuilder":
        """Mark parameters as trainable (optimized via Adam).

        Trainable parameters are optimized alongside model parameters during
        training, rather than being fixed constants. This is useful for
        eigenvalue problems where E should be learned.

        Args:
            *names: Parameter names to make trainable (e.g., "E").

        Returns:
            Self for chaining.

        Example:
            >>> PDEBuilder("Derivative(psi, x, 2)/2 + E*psi = 0")
            ...     .params(E=1.0)           # Initial guess
            ...     .trainable_params("E")   # Train E via Adam
            ...     .loss(EigenvalueLoss(method="trainable"))
            ...     .solve()
        """
        self._trainable_param_names.update(names)
        return self

    def function(self, name: str, fn: Callable) -> "PDEBuilder":
        """Register a custom function for use in the equation.

        Custom functions allow you to define arbitrary potentials, forcing
        terms, or other functions that appear in your PDE.

        Args:
            name: Function name as it appears in the equation (e.g., "V", "f").
            fn: Callable taking mx.array and returning mx.array.

        Returns:
            Self for chaining.

        Example:
            # Quartic potential for anharmonic oscillator
            >>> PDEBuilder("-Derivative(psi,x,2)/2 + V(x)*psi = E*psi")
            ...     .function("V", lambda x: x**4)
            ...     .params(E=1.0)
            ...     .solve()

            # Custom forcing term
            >>> PDEBuilder("Derivative(u,x,2) = f(x)")
            ...     .function("f", lambda x: mx.sin(2*mx.pi*x))
            ...     .solve()

            # Multiple custom functions
            >>> PDEBuilder("-Derivative(psi,x,2)/2 + V(x)*psi + W(x)*psi = E*psi")
            ...     .function("V", lambda x: x**2)    # Harmonic
            ...     .function("W", lambda x: 0.1*x**4) # Perturbation
            ...     .solve()
        """
        self._custom_functions[name.lower()] = fn
        return self

    def derivative_method(self, method: str = "autodiff", h: float = 1e-4) -> "PDEBuilder":
        """Configure derivative computation method.

        Args:
            method: Method for computing derivatives:
                - "autodiff" (default): Automatic differentiation. Accurate and recommended.
                - "finite_diff": Finite differences (legacy). Less accurate due to float32 cancellation.
            h: Step size for finite differences (only used if method="finite_diff").

        Returns:
            Self for chaining.

        Example:
            # Use autodiff (recommended, default)
            >>> PDEBuilder("-Derivative(psi,x,2)/2 + x**2*psi/2 = E*psi")
            ...     .derivative_method("autodiff")  # Explicit, but this is the default
            ...     .solve()

            # Use finite differences (legacy, not recommended)
            >>> PDEBuilder("-Derivative(psi,x,2)/2 + x**2*psi/2 = E*psi")
            ...     .derivative_method("finite_diff", h=1e-4)
            ...     .solve()

        Note:
            Autodiff is significantly more accurate than finite differences for float32 precision.
            Finite differences suffer from catastrophic cancellation in second derivatives.
            For example, on f''(2) for f(x)=x³ (analytical=12.0):
            - Autodiff: 12.00 (correct)
            - Finite diff float32: -47.68 (wrong due to cancellation)
        """
        if method not in ("autodiff", "finite_diff"):
            raise ValueError(f"Unknown derivative method: {method}. Use 'autodiff' or 'finite_diff'.")

        if method == "finite_diff":
            import warnings
            warnings.warn(
                "finite_diff is deprecated (less accurate). Use autodiff for better accuracy.",
                DeprecationWarning,
                stacklevel=2
            )

        self._derivative_method = method
        self._finite_diff_h = h
        return self

    def live_plot(
        self,
        freq: int = 500,
        x_range=None,
        analytic_fn=None,
        n_plot_points: int = 200,
        show_kan: bool = False,
        var_name: str = "x",
        fn_name: str = "W",
        figsize: tuple = (8, 4),
        # Video export
        record: bool = False,
        video_path: str = "training.mp4",
        fps: int = 15,
        dpi: int = 150,
        writer: str = "ffmpeg",
    ) -> "PDEBuilder":
        """Enable live visualization of the learned function during training.

        Spawns a matplotlib figure that updates every `freq` logged steps,
        showing the current learned function alongside an optional analytic
        reference. Works in both Jupyter notebooks and terminal scripts.

        Set ``record=True`` to export the entire training as an MP4 video
        (requires ffmpeg) or GIF (requires Pillow, use ``writer="pillow"``).

        Args:
            freq: Update the plot every this many logged steps (default 500).
                  Effective refresh is max(freq, phase.log_freq).
            x_range: (lo, hi) for the plot x-axis. Inferred from domain if None.
            analytic_fn: Optional ``f(x_np) -> y_np`` for a reference dashed line.
            n_plot_points: Number of x-points to evaluate the model on.
            show_kan: If True, call ``model.plot()`` at each phase end.
            var_name: x-axis label (e.g. "r²").
            fn_name: y-axis / legend label (e.g. "W").
            figsize: ``(width, height)`` in inches.
            record: If True, capture every plot frame and write a video on close.
            video_path: Output path (e.g. "training.mp4" or "training.gif").
            fps: Frames per second in the output video (default 15).
            dpi: Dots-per-inch for video frames (default 150).
            writer: ``"ffmpeg"`` (MP4, needs ffmpeg binary) or
                    ``"pillow"`` (GIF, needs Pillow).  Falls back to
                    ``"pillow"`` automatically if ffmpeg is unavailable.

        Returns:
            Self for chaining.

        Examples::

            # Live plot only
            .live_plot(freq=200, analytic_fn=lambda r2: np.exp(-r2)/np.pi,
                       var_name="r²", fn_name="W")

            # Export MP4
            .live_plot(freq=100, record=True, video_path="wigner_n0.mp4", fps=20)

            # Export GIF (no ffmpeg needed)
            .live_plot(freq=100, record=True, video_path="wigner_n0.gif",
                       writer="pillow", fps=10)
        """
        from ..visualization import LivePlotter
        self._live_plotter = LivePlotter(
            freq=freq,
            x_range=x_range,
            analytic_fn=analytic_fn,
            n_plot_points=n_plot_points,
            show_kan=show_kan,
            var_name=var_name,
            fn_name=fn_name,
            figsize=figsize,
            record=record,
            video_path=video_path,
            fps=fps,
            dpi=dpi,
            writer=writer,
        )
        return self

    def domain(self, d: Union[List, Tuple, Dict, Domain]) -> "PDEBuilder":
        """Set the problem domain.

        Args:
            d: Domain specification.

        Returns:
            Self for chaining.
        """
        self._domain = d
        return self

    def bc(self, *conditions) -> "PDEBuilder":
        """Add boundary conditions.

        Args:
            *conditions: Boundary condition specifications.

        Returns:
            Self for chaining.
        """
        self._bcs.extend(conditions)
        return self

    def loss(self, term: LossTerm) -> "PDEBuilder":
        """Add a loss term.

        Args:
            term: Loss term to add.

        Returns:
            Self for chaining.
        """
        self._losses.append(term)
        return self

    def losses(self, terms: List[LossTerm]) -> "PDEBuilder":
        """Add multiple loss terms.

        Args:
            terms: List of loss terms.

        Returns:
            Self for chaining.
        """
        self._losses.extend(terms)
        return self

    def phase(
        self,
        name: str,
        steps: int = 1000,
        lr: float = 0.01,
        *,
        loss_weights: Optional[Dict[str, float]] = None,
        n_points: int = 1000,
        grid_update_before: bool = False,
        grid_update_after: bool = False,
        prune_before: bool = False,
        prune_after: bool = False,
        prune_threshold: float = 0.01,
        symbolic_after: bool = False,
        symbolic_threshold: float = 0.95,
        log_freq: int = 100,
        **kwargs
    ) -> "PDEBuilder":
        """Add a training phase.

        Args:
            name: Phase name.
            steps: Number of training steps.
            lr: Learning rate for this phase.
            loss_weights: Override weights for specific loss terms.
            n_points: Number of collocation points per batch.
            grid_update_before: Update grid before this phase.
            grid_update_after: Update grid after this phase.
            prune_before: Prune model before this phase.
            prune_after: Prune model after this phase.
            prune_threshold: Threshold for pruning (0.01 = remove edges < 1% contribution).
            symbolic_after: Extract symbolic formulas after this phase.
            symbolic_threshold: R² threshold for auto_symbolic (default 0.95).
            log_freq: How often to log progress.
            **kwargs: Additional phase configuration.

        Returns:
            Self for chaining.

        Example:
            >>> builder.phase("refine", steps=500, lr=0.001,
            ...               prune_after=True, prune_threshold=0.01,
            ...               symbolic_after=True, symbolic_threshold=0.95)
        """
        self._phases.append(TrainingPhase(
            name=name,
            steps=steps,
            lr=lr,
            loss_weights=loss_weights or {},
            n_points=n_points,
            grid_update_before=grid_update_before,
            grid_update_after=grid_update_after,
            prune_before=prune_before,
            prune_after=prune_after,
            prune_threshold=prune_threshold,
            symbolic_after=symbolic_after,
            symbolic_threshold=symbolic_threshold,
            log_freq=log_freq,
            **kwargs
        ))
        return self

    def schedule(self, s: TrainingSchedule) -> "PDEBuilder":
        """Set the full training schedule.

        Args:
            s: TrainingSchedule object.

        Returns:
            Self for chaining.
        """
        self._phases = s.phases.copy()
        return self

    def model(self, **config) -> "PDEBuilder":
        """Configure the model.

        Args:
            **config: Model configuration (width, basis, grid, k, etc.).

        Returns:
            Self for chaining.
        """
        self._model_config.update(config)
        return self

    def prune(self, after_phase: Optional[str] = None, threshold: float = 0.01) -> "PDEBuilder":
        """Enable pruning after a specific phase or the last phase.

        Args:
            after_phase: Name of phase after which to prune. If None, adds pruning
                        to the last phase (or creates a pruning phase).
            threshold: Pruning threshold (0.01 = remove edges < 1% contribution).

        Returns:
            Self for chaining.

        Example:
            >>> builder.phase("train", steps=1000, lr=0.01).prune(threshold=0.01)
        """
        if after_phase is None:
            # Apply to last phase
            if self._phases:
                self._phases[-1].prune_after = True
                self._phases[-1].prune_threshold = threshold
            else:
                # No phases yet - add a prune-only phase marker
                self._model_config['_prune_after_train'] = True
                self._model_config['_prune_threshold'] = threshold
        else:
            # Find and update the specified phase
            for phase in self._phases:
                if phase.name == after_phase:
                    phase.prune_after = True
                    phase.prune_threshold = threshold
                    break
        return self

    def symbolic(self, after_phase: Optional[str] = None, threshold: float = 0.95) -> "PDEBuilder":
        """Enable symbolic extraction after a specific phase or the last phase.

        Args:
            after_phase: Name of phase after which to extract. If None, adds
                        symbolic extraction to the last phase.
            threshold: R² threshold for auto_symbolic.

        Returns:
            Self for chaining.

        Example:
            >>> builder.phase("train", steps=1000, lr=0.01).symbolic(threshold=0.95)
        """
        if after_phase is None:
            # Apply to last phase
            if self._phases:
                self._phases[-1].symbolic_after = True
                self._phases[-1].symbolic_threshold = threshold
            else:
                # No phases yet - add marker
                self._model_config['_symbolic_after_train'] = True
                self._model_config['_symbolic_threshold'] = threshold
        else:
            # Find and update the specified phase
            for phase in self._phases:
                if phase.name == after_phase:
                    phase.symbolic_after = True
                    phase.symbolic_threshold = threshold
                    break
        return self

    def verbose(self, v: bool = True) -> "PDEBuilder":
        """Set verbosity.

        Args:
            v: Whether to print progress.

        Returns:
            Self for chaining.
        """
        self._verbose = v
        return self

    def quiet(self) -> "PDEBuilder":
        """Disable verbose output.

        Returns:
            Self for chaining.
        """
        self._verbose = False
        return self

    # =========================================================================
    # NEW TIER 1 FEATURES
    # =========================================================================

    def preset(self, name: str, **overrides) -> "PDEBuilder":
        """Use a physics preset for model configuration.

        Presets provide optimized configurations for common physics problems:
        - quantum_oscillator: Hermite basis for harmonic oscillator wavefunctions
        - wave_equation: Fourier basis for periodic/oscillatory solutions
        - spectral: Chebyshev basis for high-accuracy spectral methods
        - radial: Laguerre basis for radial problems (hydrogen atom)
        - angular: Legendre basis for angular momentum problems
        - general: B-spline basis for general function approximation

        Args:
            name: Preset name (e.g., "quantum_oscillator", "wave_equation").
            **overrides: Override any preset parameter.

        Returns:
            Self for chaining.

        Example:
            >>> PDEBuilder("schrodinger").preset("quantum_oscillator").solve()
            >>> PDEBuilder("wave").preset("wave_equation", basis_M=31).solve()
        """
        from ..presets import PRESETS, get_preset_config

        if name not in PRESETS:
            available = ", ".join(sorted(PRESETS.keys()))
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")

        self._preset_name = name
        # Get preset config and merge with overrides
        config = get_preset_config(name)
        config.update(overrides)
        self._model_config.update(config)
        return self

    def optimizer(
        self,
        type: Literal["adam", "lbfgs", "sgd", "adamw"] = "adam",
        **kwargs
    ) -> "PDEBuilder":
        """Configure the optimizer.

        Args:
            type: Optimizer type. Options:
                - "adam": Adam optimizer (default, good for most cases)
                - "lbfgs": L-BFGS quasi-Newton (often faster convergence for physics)
                - "sgd": Stochastic gradient descent with momentum
                - "adamw": Adam with decoupled weight decay
            **kwargs: Optimizer-specific parameters:
                - adam/adamw: lr, beta1, beta2, eps, weight_decay
                - lbfgs: max_iter, tolerance_grad, tolerance_change, history_size
                - sgd: lr, momentum

        Returns:
            Self for chaining.

        Example:
            >>> PDEBuilder("schrodinger").optimizer("lbfgs", max_iter=100).solve()
            >>> PDEBuilder("heat").optimizer("adam", lr=0.001).solve()
        """
        self._optimizer_config = {"type": type.lower(), **kwargs}
        return self

    def checkpoint(self, path: str) -> "PDEBuilder":
        """Save model checkpoint after training.

        The checkpoint will be saved after all training phases complete.
        Use `.restore()` to load from a checkpoint.

        Args:
            path: Path to save checkpoint (e.g., "model.ckpt").

        Returns:
            Self for chaining.

        Example:
            >>> PDEBuilder("schrodinger").checkpoint("schrodinger.ckpt").solve()
        """
        self._checkpoint_path = path
        return self

    def restore(self, path: str) -> "PDEBuilder":
        """Restore model from checkpoint before training.

        This loads a previously saved model state. Training continues
        from this restored state.

        Args:
            path: Path to checkpoint file.

        Returns:
            Self for chaining.

        Example:
            >>> PDEBuilder("schrodinger").restore("schrodinger.ckpt").solve()
        """
        # Store path for later loading when model is created
        self._model_config['_restore_from'] = path
        return self

    def visualize(self, after_phase: Optional[str] = None) -> "PDEBuilder":
        """Enable visualization after training or specific phase.

        Args:
            after_phase: Phase name after which to visualize.
                        If None, visualizes after all training.

        Returns:
            Self for chaining.

        Example:
            >>> PDEBuilder("schrodinger").visualize().solve()
            >>> PDEBuilder("wave").visualize("refine").solve()
        """
        if after_phase is not None:
            self._visualize_phases.append(after_phase)
        else:
            self._visualize_phases.append("__final__")
        return self

    def visualize_after(self, phase_name: str) -> "PDEBuilder":
        """Enable visualization after a specific phase.

        Alias for `.visualize(after_phase=phase_name)`.

        Args:
            phase_name: Phase name after which to visualize.

        Returns:
            Self for chaining.
        """
        return self.visualize(after_phase=phase_name)

    def plot(self, **kwargs) -> None:
        """Plot the trained model's activation functions.

        Must be called after `.solve()`.

        Args:
            **kwargs: Arguments passed to model.plot().

        Raises:
            RuntimeError: If called before `.solve()`.

        Example:
            >>> builder = PDEBuilder("schrodinger")
            >>> model, history = builder.solve()
            >>> builder.plot()
        """
        if self._trained_model is None:
            raise RuntimeError("No trained model. Call .solve() first.")

        if hasattr(self._trained_model, 'plot'):
            self._trained_model.plot(**kwargs)
        else:
            print("Model does not support plotting.")

    def plot_activations(self, x: Optional[mx.array] = None, **kwargs) -> None:
        """Plot activation distributions for the trained model.

        Must be called after `.solve()`.

        Args:
            x: Input data for activations. If None, samples from domain.
            **kwargs: Arguments passed to model.plot_activations().

        Raises:
            RuntimeError: If called before `.solve()`.
        """
        if self._trained_model is None:
            raise RuntimeError("No trained model. Call .solve() first.")

        if x is None and self._domain is not None:
            domain_obj = _build_domain(self._domain, set())
            x = domain_obj.sample(1000, strategy="uniform")

        if hasattr(self._trained_model, 'plot_activations'):
            self._trained_model.plot_activations(x, **kwargs)
        else:
            print("Model does not support activation plotting.")

    def solve(
        self,
        verbose: Optional[bool] = None,
        quick: bool = False,
    ) -> Tuple[Any, TrainingHistory]:
        """Execute the solve.

        Supports sensible defaults for one-liner usage. If no phases, losses,
        or model config are set, reasonable defaults are applied based on the
        equation type.

        Args:
            verbose: Whether to print progress. If None, uses the builder's setting.
            quick: If True, use minimal training for fast iteration (100 steps,
                   no grid updates). Useful in REPL for quick testing.

        Returns:
            Tuple of (trained_model, training_history).

        Example:
            # One-liner with sensible defaults
            >>> model, h = PDEBuilder("Derivative(u,x,2) + u = 0").solve()

            # Quick mode for fast iteration in REPL
            >>> model, h = PDEBuilder("Derivative(u,x,2) + u = 0").solve(quick=True)
        """
        import warnings

        # Warn about unimplemented optimizers
        opt_type = self._optimizer_config.get('type', 'adam')
        if opt_type in ('sgd', 'adamw'):
            warnings.warn(
                f"Optimizer '{opt_type}' is not yet implemented. Using Adam instead. "
                f"SGD and AdamW support is planned for a future release.",
                UserWarning
            )

        # Apply quick mode settings
        if quick:
            if not self._phases:
                self._phases = [TrainingPhase(
                    name="quick",
                    steps=100,
                    lr=0.01,
                    n_points=500,
                    log_freq=25,
                )]
            self._verbose = False if verbose is None else verbose

        # Build schedule from phases if provided
        schedule = None
        if self._phases:
            schedule = TrainingSchedule(phases=self._phases)

        # Use provided verbose or fall back to builder setting
        v = verbose if verbose is not None else self._verbose

        # Add optimizer config to model_config for pde() to use
        model_config = self._model_config.copy() if self._model_config else {}
        model_config['_optimizer_config'] = self._optimizer_config
        model_config['_visualize_phases'] = self._visualize_phases
        model_config['_custom_functions'] = self._custom_functions
        model_config['_derivative_method'] = self._derivative_method
        model_config['_finite_diff_h'] = self._finite_diff_h
        if self._live_plotter is not None:
            model_config['_live_plotter'] = self._live_plotter

        model, history = pde(
            equation=self._equation,
            domain=self._domain,
            params=self._params,
            trainable_param_names=self._trainable_param_names if self._trainable_param_names else None,
            bc=self._bcs[0] if len(self._bcs) == 1 and isinstance(self._bcs[0], str) else self._bcs,
            schedule=schedule,
            losses=self._losses if self._losses else None,
            model_config=model_config,
            verbose=v,
        )

        # Store trained model for visualization methods
        self._trained_model = model

        # Close live plotter (leaves final figure open)
        if self._live_plotter is not None:
            self._live_plotter.close()

        # Save checkpoint if requested. Persist trainable params (e.g. the
        # eigenvalue E) as `extra` so a reload can recover them.
        if self._checkpoint_path is not None:
            extra = None
            tp = getattr(history, "trainable_params", None)
            if tp:
                extra = {"trainable_params": {k: float(np.array(v)) for k, v in tp.items()}}
            self._save_checkpoint(model, self._checkpoint_path, v, extra=extra)

        # Final visualization if requested
        if "__final__" in self._visualize_phases:
            self._do_visualize(model, v)

        return model, history

    def _save_checkpoint(
        self, model: Any, path: str, verbose: bool, extra: Optional[dict] = None
    ) -> None:
        """Save model checkpoint.

        Args:
            model: Trained model.
            path: Checkpoint path.
            verbose: Whether to print info.
            extra: Optional extra data (e.g. trainable params) to persist.
        """
        if verbose:
            print(f"Saving checkpoint to {path}...")

        if hasattr(model, 'saveckpt'):
            model.saveckpt(path, extra=extra)
            if verbose:
                print(f"Checkpoint saved: {path}")
        else:
            # Fallback: save state dict using mlx
            import pickle
            from pathlib import Path

            ckpt_path = Path(path)
            state = {
                'width': getattr(model, 'width', None),
                'basis': getattr(model, 'basis', None),
            }

            # Save layer parameters
            if hasattr(model, 'layers'):
                state['layers'] = []
                for layer in model.layers:
                    layer_state = {}
                    if hasattr(layer, 'coef'):
                        layer_state['coef'] = np.array(layer.coef)
                    if hasattr(layer, 'scale_sp'):
                        layer_state['scale_sp'] = np.array(layer.scale_sp)
                    if hasattr(layer, 'scale_base'):
                        layer_state['scale_base'] = np.array(layer.scale_base)
                    if hasattr(layer, 'grid') and layer.grid is not None:
                        layer_state['grid'] = np.array(layer.grid)
                    state['layers'].append(layer_state)

            with open(ckpt_path, 'wb') as f:
                pickle.dump(state, f)

            if verbose:
                print(f"Checkpoint saved (fallback): {path}")

    def _do_visualize(self, model: Any, verbose: bool) -> None:
        """Visualize the model.

        Args:
            model: Model to visualize.
            verbose: Whether to print info.
        """
        if verbose:
            print("Generating visualization...")

        if hasattr(model, 'plot'):
            try:
                model.plot()
            except Exception as e:
                if verbose:
                    print(f"Visualization failed: {e}")

    def analyze(self) -> PDEAnalysis:
        """Analyze the equation without solving.

        Returns:
            PDEAnalysis object with equation information.
        """
        # Handle template names
        equation = self._equation
        equation_lower = equation.lower().replace('-', '_').replace(' ', '_')
        if equation_lower in TEMPLATES:
            equation = TEMPLATES[equation_lower]

        parsed = parse(equation, format=self._format)
        return analyze(parsed)

    def summary(self, print_it: bool = True) -> str:
        """Get a summary of the problem configuration.

        REPL-friendly: prints by default, also returns the string.

        Args:
            print_it: If True (default), prints the summary. Set False to just return.

        Returns:
            Human-readable summary string.

        Example:
            >>> b = PDEBuilder("Derivative(u,x,2) + u = 0").domain([0, 3.14])
            >>> b.summary()
            PDEBuilder Configuration
            ─────────────────────────
            Equation: Derivative(u,x,2) + u = 0
            ...
        """
        analysis = self.analyze()

        # Build formatted summary
        lines = [
            "PDEBuilder Configuration",
            "─" * 25,
            f"  Equation: {self._equation}",
            f"  Type: {analysis.equation_type.name}",
            f"  Unknowns: {', '.join(analysis.unknowns)}",
            f"  Coordinates: {', '.join(analysis.coordinates) or 'auto'}",
        ]

        # Parameters
        params = self._params or analysis.parameters
        if params:
            params_str = ", ".join(f"{k}={v}" for k, v in params.items())
            lines.append(f"  Params: {{{params_str}}}")
        else:
            lines.append("  Params: (none)")

        # Trainable params
        if self._trainable_param_names:
            lines.append(f"  Trainable: {list(self._trainable_param_names)}")

        # Domain
        if self._domain:
            if isinstance(self._domain, (list, tuple)) and len(self._domain) == 2:
                if isinstance(self._domain[0], (int, float)):
                    lines.append(f"  Domain: [{self._domain[0]}, {self._domain[1]}]")
                else:
                    lines.append(f"  Domain: {self._domain}")
            else:
                lines.append(f"  Domain: {self._domain}")
        else:
            lines.append("  Domain: (not set, defaults to [-5, 5])")

        # Losses
        if self._losses:
            loss_names = [type(l).__name__ for l in self._losses]
            if len(loss_names) <= 3:
                lines.append(f"  Losses: {', '.join(loss_names)}")
            else:
                lines.append(f"  Losses: {len(self._losses)} terms ({', '.join(loss_names[:2])}, ...)")
        else:
            lines.append("  Losses: (default)")

        # Phases
        if self._phases:
            phase_info = []
            for p in self._phases:
                phase_info.append(f"{p.name}({p.steps} steps)")
            lines.append(f"  Phases: {', '.join(phase_info)}")
        else:
            lines.append("  Phases: (default schedule)")

        # Model config
        if self._model_config:
            width = self._model_config.get('width')
            if width:
                lines.append(f"  Model: width={width}")
            basis = self._model_config.get('basis')
            if basis:
                lines.append(f"         basis={basis}")
        else:
            lines.append("  Model: (default)")

        # Optimizer
        if self._optimizer_config.get('type', 'adam') != 'adam':
            lines.append(f"  Optimizer: {self._optimizer_config['type']}")

        result = '\n'.join(lines)

        if print_it:
            print(result)

        return result

    def dry_run(self) -> "PDEBuilder":
        """Validate configuration without training.

        Parses the equation, builds the domain, and checks for issues.
        Useful for quick iteration in REPL before committing to training.

        Returns:
            Self for chaining.

        Example:
            >>> b = PDEBuilder("Derivative(u,x,2) + u = 0").domain([0, 3.14])
            >>> b.dry_run()  # Validates and prints summary
            >>> b.solve()    # Now train
        """
        import warnings

        print("Dry Run: Validating configuration...")
        print()

        issues = []

        # 1. Parse equation
        try:
            analysis = self.analyze()
            print(f"✓ Equation parsed: {analysis.equation_type.name}")
        except Exception as e:
            issues.append(f"✗ Equation parse error: {e}")
            print(issues[-1])
            return self

        # 2. Check trainable params exist in params
        for name in self._trainable_param_names:
            if name not in self._params:
                issues.append(f"⚠ Trainable param '{name}' not in .params() - will use default")

        # 3. Check domain
        if self._domain is None:
            print("⚠ No domain set, using default [-5, 5]")
        else:
            try:
                domain_obj = _build_domain(self._domain, analysis.coordinates)
                print(f"✓ Domain: {domain_obj.dim}D, bounds valid")
            except Exception as e:
                issues.append(f"✗ Domain error: {e}")

        # 4. Check phases
        if not self._phases:
            print("⚠ No phases set, using default schedule")
        else:
            total_steps = sum(p.steps for p in self._phases)
            print(f"✓ Training: {len(self._phases)} phases, {total_steps} total steps")

        # 5. Check losses
        if not self._losses:
            if analysis.is_eigenvalue_problem:
                print("✓ Using default eigenvalue losses")
            else:
                print("✓ Using default PINN losses")
        else:
            print(f"✓ Custom losses: {len(self._losses)} terms")

        # 6. Check optimizer
        opt_type = self._optimizer_config.get('type', 'adam')
        if opt_type in ('sgd', 'adamw'):
            issues.append(f"⚠ Optimizer '{opt_type}' not fully implemented, may use Adam")

        # Print issues
        if issues:
            print()
            print("Issues found:")
            for issue in issues:
                print(f"  {issue}")
        else:
            print()
            print("✓ Configuration looks good!")

        print()
        self.summary()

        return self


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _build_domain(
    domain_spec: Optional[Union[List, Tuple, Dict, Domain]],
    coordinates: Set[str]
) -> Domain:
    """Build Domain object from specification."""
    if domain_spec is None:
        # Default domain based on coordinates
        dim = max(1, len(coordinates))
        return Domain.interval(-5, 5) if dim == 1 else Domain([(-5, 5)] * dim)

    if isinstance(domain_spec, Domain):
        return domain_spec

    if isinstance(domain_spec, dict):
        # Named coordinates: {"x": [-1, 1], "t": [0, 2]}
        bounds = [tuple(v) for v in domain_spec.values()]
        return Domain(bounds)

    if isinstance(domain_spec, (list, tuple)):
        # Check if it's a single interval [a, b] or list of bounds
        if len(domain_spec) == 2 and isinstance(domain_spec[0], (int, float)):
            return Domain.interval(domain_spec[0], domain_spec[1])
        else:
            # List of bounds: [(a1, b1), (a2, b2), ...]
            return Domain([tuple(b) for b in domain_spec])

    return Domain.interval(-5, 5)


def _build_loss_composer(
    losses: Optional[List[LossTerm]],
    analysis: PDEAnalysis
) -> LossComposer:
    """Build loss composer from specification or defaults."""
    composer = LossComposer()

    if losses:
        # Use provided losses
        for loss in losses:
            composer.add(loss)
    else:
        # Use defaults based on equation type
        composer.add(PDEResidualLoss(weight=1.0))
        composer.add(BoundaryConditionLoss(weight=10.0))

        if analysis.is_eigenvalue_problem:
            composer.add(NormalizationLoss(weight=10.0))
            composer.add(NonTrivialLoss(weight=20.0))
            composer.add(EigenvalueLoss(weight=1.0, method="rayleigh"))

    return composer


def _create_model(
    config: Optional[Dict],
    domain: Domain,
    analysis: PDEAnalysis
) -> Any:
    """Create MultKAN model from configuration."""
    from ..multkan import MultKAN

    config = config or {}

    # Check for restore path
    restore_path = config.pop('_restore_from', None)

    # Remove internal config keys
    config.pop('_optimizer_config', None)
    config.pop('_visualize_phases', None)
    config.pop('_prune_after_train', None)
    config.pop('_prune_threshold', None)
    config.pop('_symbolic_after_train', None)
    config.pop('_symbolic_threshold', None)
    config.pop('_custom_functions', None)
    config.pop('_live_plotter', None)

    # Default width based on domain dimension
    default_width = [domain.dim, 20, 20, 1]
    width = config.get('width', default_width)

    # Adjust input dimension if needed
    if width[0] != domain.dim:
        width = [domain.dim] + list(width[1:])

    # Default basis from analysis
    basis = config.get('basis', analysis.recommended_basis)

    # Other defaults
    grid = config.get('grid', 10)
    k = config.get('k', 3)

    # Extract basis_kwargs
    basis_kwargs = config.get('basis_kwargs', {})
    basis_M = config.get('basis_M', 8)

    # Resolve base_fun: accept callable or string name
    _base_fun_raw = config.get('base_fun', nn.silu)
    if isinstance(_base_fun_raw, str):
        _base_fun_map = {
            'silu': nn.silu,
            'identity': lambda x: x,
            'tanh': mx.tanh,
            'relu': nn.relu,
            'gelu': nn.gelu,
        }
        if _base_fun_raw not in _base_fun_map:
            raise ValueError(
                f"Unknown base_fun string '{_base_fun_raw}'. "
                f"Valid options: {list(_base_fun_map)}"
            )
        base_fun = _base_fun_map[_base_fun_raw]
    else:
        base_fun = _base_fun_raw

    noise_scale = config.get('noise_scale', 0.1)
    seed = config.get('seed', None)
    mult_arity = config.get('mult_arity', 2)
    basis_per_mult_slot = config.get('basis_per_mult_slot', None)
    grid_range = config.get('grid_range', (-1.0, 1.0))

    # Create model
    model = MultKAN(
        width=width,
        basis=basis,
        basis_M=basis_M,
        basis_kwargs=basis_kwargs,
        grid=grid,
        k=k,
        base_fun=base_fun,
        noise_scale=noise_scale,
        seed=seed,
        mult_arity=mult_arity,
        basis_per_mult_slot=basis_per_mult_slot,
        grid_range=grid_range,
    )

    # Restore from checkpoint if path provided
    if restore_path is not None:
        _restore_model(model, restore_path)

    return model


def _restore_model(model: Any, path: str) -> None:
    """Restore model from checkpoint.

    Args:
        model: Model to restore into.
        path: Checkpoint path.
    """
    if hasattr(model, 'loadckpt'):
        model.loadckpt(path)
    else:
        # Fallback: load state dict
        import pickle
        from pathlib import Path

        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(ckpt_path, 'rb') as f:
            state = pickle.load(f)

        # Restore layer parameters
        if 'layers' in state and hasattr(model, 'layers'):
            for i, layer_state in enumerate(state['layers']):
                if i < len(model.layers):
                    layer = model.layers[i]
                    if 'coef' in layer_state:
                        layer.coef = mx.array(layer_state['coef'])
                    if 'scale_sp' in layer_state:
                        layer.scale_sp = mx.array(layer_state['scale_sp'])
                    if 'scale_base' in layer_state:
                        layer.scale_base = mx.array(layer_state['scale_base'])


# =============================================================================
# CONVENIENCE SHORTCUTS
# =============================================================================

def schrodinger(
    E: float = 0.5,
    domain: Optional[Union[List, Domain]] = None,
    V: Optional[Callable] = None,
    **kwargs
) -> Tuple[Any, TrainingHistory]:
    """Shortcut for Schrödinger equation.

    Args:
        E: Energy eigenvalue.
        domain: Domain (default: [-5, 5]).
        V: Potential function (default: harmonic oscillator).
        **kwargs: Additional arguments to pde().

    Returns:
        (model, history) tuple.
    """
    domain = domain or [-5, 5]
    return pde("schrodinger", domain=domain, params={"E": E}, **kwargs)


def wheeler_dewitt(
    theta: float = 0.0,
    domain: Optional[Union[List, Domain]] = None,
    **kwargs
) -> Tuple[Any, TrainingHistory]:
    """Shortcut for Wheeler-DeWitt equation.

    Args:
        theta: Deformation parameter (0 for standard WDW).
        domain: Domain (default: [0.1, 5]).
        **kwargs: Additional arguments to pde().

    Returns:
        (model, history) tuple.
    """
    domain = domain or [0.1, 5]
    if theta > 0:
        return pde("Ĥ ⋆_θ Ψ = 0", domain=domain, params={"theta": theta}, **kwargs)
    return pde("wheeler_dewitt", domain=domain, **kwargs)


def klein_gordon(
    m: float = 0.0,
    domain: Optional[Union[List, Dict, Domain]] = None,
    **kwargs
) -> Tuple[Any, TrainingHistory]:
    """Shortcut for Klein-Gordon equation.

    Args:
        m: Mass parameter.
        domain: Domain (default: spacetime).
        **kwargs: Additional arguments to pde().

    Returns:
        (model, history) tuple.
    """
    domain = domain or {"t": [0, 5], "x": [-5, 5]}
    return pde("klein_gordon", domain=domain, params={"m": m}, **kwargs)


def wave(
    c: float = 1.0,
    domain: Optional[Union[List, Dict, Domain]] = None,
    **kwargs
) -> Tuple[Any, TrainingHistory]:
    """Shortcut for wave equation.

    Args:
        c: Wave speed.
        domain: Domain (default: spacetime).
        **kwargs: Additional arguments to pde().

    Returns:
        (model, history) tuple.
    """
    domain = domain or {"t": [0, 5], "x": [-5, 5]}
    return pde("wave", domain=domain, params={"c": c}, **kwargs)


def heat(
    alpha: float = 1.0,
    domain: Optional[Union[List, Dict, Domain]] = None,
    **kwargs
) -> Tuple[Any, TrainingHistory]:
    """Shortcut for heat equation.

    Args:
        alpha: Thermal diffusivity.
        domain: Domain (default: spacetime).
        **kwargs: Additional arguments to pde().

    Returns:
        (model, history) tuple.
    """
    domain = domain or {"t": [0, 1], "x": [-1, 1]}
    return pde("heat", domain=domain, params={"alpha": alpha}, **kwargs)


def laplace(
    domain: Optional[Union[List, Dict, Domain]] = None,
    **kwargs
) -> Tuple[Any, TrainingHistory]:
    """Shortcut for Laplace equation.

    Args:
        domain: Domain (default: 2D square).
        **kwargs: Additional arguments to pde().

    Returns:
        (model, history) tuple.
    """
    domain = domain or {"x": [-1, 1], "y": [-1, 1]}
    return pde("laplace", domain=domain, **kwargs)


def poisson(
    f: Optional[Callable] = None,
    domain: Optional[Union[List, Dict, Domain]] = None,
    **kwargs
) -> Tuple[Any, TrainingHistory]:
    """Shortcut for Poisson equation.

    Args:
        f: Source function.
        domain: Domain (default: 2D square).
        **kwargs: Additional arguments to pde().

    Returns:
        (model, history) tuple.
    """
    domain = domain or {"x": [-1, 1], "y": [-1, 1]}
    return pde("poisson", domain=domain, **kwargs)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def list_equations() -> Dict[str, str]:
    """List all available equation templates.

    Returns:
        Dict mapping template names to equation strings.
    """
    return TEMPLATES.copy()


def help_equation(name: str) -> str:
    """Get help for a specific equation template.

    Args:
        name: Template name.

    Returns:
        Help string with equation details.
    """
    name_lower = name.lower().replace('-', '_').replace(' ', '_')
    if name_lower not in TEMPLATES:
        available = ', '.join(sorted(TEMPLATES.keys()))
        return f"Unknown equation: {name}. Available: {available}"

    eq = TEMPLATES[name_lower]
    parsed = parse(eq)
    analysis = analyze(parsed)

    help_text = f"""
{name}
{'='*len(name)}

Equation: {eq}

Type: {analysis.equation_type.name} ({analysis.equation_type.category})
Unknown: {', '.join(analysis.unknowns)}
Coordinates: {', '.join(analysis.coordinates)}
Parameters: {analysis.parameters}

Eigenvalue problem: {analysis.is_eigenvalue_problem}
Time-dependent: {analysis.is_time_dependent}
Recommended basis: {analysis.recommended_basis}

Usage:
    model, history = pde("{name}", domain=[...], params={{...}})

Or with builder:
    model, history = PDEBuilder("{name}").domain([...]).params(...).solve()
"""
    return help_text


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main functions
    'pde',
    'PDEBuilder',

    # Shortcuts
    'schrodinger',
    'wheeler_dewitt',
    'klein_gordon',
    'wave',
    'heat',
    'laplace',
    'poisson',

    # Utilities
    'list_equations',
    'help_equation',
    'list_presets',
    'describe_preset',
    'parse',
    'analyze',

    # Loss terms
    'LossTerm',
    'PDEResidualLoss',
    'BoundaryConditionLoss',
    'NormalizationLoss',
    'NonTrivialLoss',
    'EigenvalueLoss',
    'SmoothnessLoss',
    'DecayLoss',
    'AnchorLoss',
    'RegularizationLoss',
    'DataLoss',
    'LossComposer',

    # Training
    'TrainingPhase',
    'TrainingSchedule',
    'TrainingHistory',
    'PDETrainer',

    # Types
    'Domain',
    'BoundaryCondition',
    'ParsedPDE',
    'PDEAnalysis',
    'CompiledResidual',
]


# =============================================================================
# PRESET UTILITIES (Re-exported)
# =============================================================================

def list_presets() -> Dict[str, str]:
    """List available physics presets with descriptions.

    Returns:
        Dict mapping preset names to descriptions.

    Example:
        >>> for name, desc in list_presets().items():
        ...     print(f"{name}: {desc}")
    """
    from ..presets import list_presets as _list_presets
    return _list_presets()


def describe_preset(name: str) -> str:
    """Get detailed description of a physics preset.

    Args:
        name: Preset name.

    Returns:
        Multi-line description of the preset.
    """
    from ..presets import describe_preset as _describe_preset
    return _describe_preset(name)
