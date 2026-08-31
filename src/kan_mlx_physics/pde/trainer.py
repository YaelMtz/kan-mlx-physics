"""Multi-Phase PDE Trainer.

Orchestrates multi-phase training for PDE problems with:
- Per-phase loss weights and learning rates
- Grid refinement between phases
- Symbolic extraction after phases
- Progress logging and history tracking

Example:
    >>> schedule = TrainingSchedule([
    ...     TrainingPhase("warmup", steps=1000, lr=0.01),
    ...     TrainingPhase("refine", steps=500, lr=0.001, grid_update_before=True),
    ...     TrainingPhase("fine-tune", steps=200, lr=0.0001, symbolic_after=True),
    ... ])

    >>> trainer = PDETrainer(problem, model, schedule)
    >>> history = trainer.train(verbose=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set, Union
import time

import mlx.core as mx
import numpy as np

from .losses import LossComposer, LossContext, LossTerm
from .compiler import CompiledResidual
from .problem import Domain


# =============================================================================
# GRADNORM LOSS BALANCER
# =============================================================================

class GradNormBalancer:
    """Automatic loss weight balancing via gradient norms.

    Implements GradNorm-style balancing (Chen et al., 2018) to automatically
    adjust loss weights so all loss terms train at similar rates.

    When some losses dominate training (high gradients) while others are
    undertrained (low gradients), GradNorm adjusts weights to balance them.

    Example:
        balancer = GradNormBalancer(["pde", "bc", "normalization"])

        for step in range(steps):
            losses, grad_norms = compute_per_loss_grads(...)
            balancer.update(losses, grad_norms)
            weights = balancer.weights  # Use for weighted sum
    """

    def __init__(
        self,
        loss_names: List[str],
        alpha: float = 1.5,
        initial_weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize GradNorm balancer.

        Args:
            loss_names: Names of loss terms to balance.
            alpha: Asymmetry parameter. Higher = more aggressive balancing.
                   1.0 = linear, 1.5 = moderate (default), 2.0 = aggressive.
            initial_weights: Starting weights (default: all 1.0).
        """
        self.loss_names = loss_names
        self.alpha = alpha

        if initial_weights is not None:
            self.weights = {name: initial_weights.get(name, 1.0) for name in loss_names}
        else:
            self.weights = {name: 1.0 for name in loss_names}

        self.initial_losses: Optional[Dict[str, float]] = None
        self._step_count = 0

    def update(
        self,
        losses: Dict[str, float],
        grad_norms: Optional[Dict[str, float]] = None,
    ) -> None:
        """Update weights based on current losses.

        Args:
            losses: Current loss values per term.
            grad_norms: Gradient norms per term (optional, uses loss ratios if None).
        """
        self._step_count += 1

        # Record initial losses on first call
        if self.initial_losses is None:
            self.initial_losses = {k: max(v, 1e-8) for k, v in losses.items()}

        # Compute inverse training rate: L(t) / L(0)
        # Higher ratio = slower training = needs more weight
        inv_rates = {}
        for name in self.loss_names:
            if name in losses and name in self.initial_losses:
                inv_rates[name] = losses[name] / self.initial_losses[name]
            else:
                inv_rates[name] = 1.0

        # Target: mean rate across all tasks
        if inv_rates:
            mean_rate = sum(inv_rates.values()) / len(inv_rates)
        else:
            return

        # Update weights: slower tasks get higher weights
        for name in self.loss_names:
            if name in inv_rates:
                # Rate ratio: how much slower/faster than average
                rate_ratio = inv_rates[name] / (mean_rate + 1e-8)
                # Adjust weight: (rate_ratio)^alpha
                # If rate_ratio > 1 (slower), weight increases
                # If rate_ratio < 1 (faster), weight decreases
                self.weights[name] = self.weights[name] * (rate_ratio ** self.alpha)

        # Renormalize to maintain total weight
        total = sum(self.weights.values())
        if total > 0:
            n_terms = len(self.weights)
            self.weights = {k: v / total * n_terms for k, v in self.weights.items()}

    def get_weight(self, name: str) -> float:
        """Get current weight for a loss term."""
        return self.weights.get(name, 1.0)

    def reset(self) -> None:
        """Reset balancer state."""
        self.initial_losses = None
        self._step_count = 0
        self.weights = {name: 1.0 for name in self.loss_names}


# =============================================================================
# TRAINING PHASE
# =============================================================================

@dataclass
class TrainingPhase:
    """Configuration for one training phase.

    Attributes:
        name: Phase name for logging.
        steps: Number of training steps.
        lr: Learning rate for this phase.
        loss_weights: Override weights for specific loss terms.
        enabled_losses: If set, only these losses are enabled.
        disabled_losses: Explicitly disable these losses.
        n_points: Number of collocation points per batch.
        n_boundary: Number of boundary points per batch.
        grid_update_before: Update grid before this phase.
        grid_update_after: Update grid after this phase.
        disable_grid_updates: Disable all grid updates for performance testing.
        grid_extend_factor: Extend grid resolution by this factor (0 = no extension).
        prune_before: Prune model before this phase.
        prune_after: Prune model after this phase.
        prune_threshold: Threshold for pruning (0.01 = remove edges < 1% contribution).
        symbolic_after: Extract symbolic formulas after this phase.
        symbolic_threshold: R² threshold for auto_symbolic (default 0.95).
        log_freq: How often to log progress.
        resample_freq: How often to resample collocation points.
        use_rbas: Use residual-based adaptive sampling (RBAS).
        rbas_weight: Weight for RBAS vs uniform sampling (0-1).
        rbas_oversample: Oversample factor for RBAS candidate pool.
        use_gradnorm: Use GradNorm-style loss balancing.
        gradnorm_alpha: Alpha parameter for GradNorm (higher = more aggressive balancing).
    """
    name: str
    steps: int
    lr: float
    loss_weights: Dict[str, float] = field(default_factory=dict)
    enabled_losses: Optional[Set[str]] = None
    disabled_losses: Optional[Set[str]] = None
    n_points: int = 1000
    n_boundary: int = 200
    grid_update_before: bool = False
    grid_update_after: bool = False
    disable_grid_updates: bool = False  # Disable all grid updates for performance testing
    grid_extend_factor: int = 0
    prune_before: bool = False
    prune_after: bool = False
    prune_threshold: float = 0.01
    symbolic_after: bool = False
    symbolic_threshold: float = 0.95
    log_freq: int = 100
    resample_freq: int = 50
    # RBAS (Residual-Based Adaptive Sampling)
    use_rbas: bool = False
    rbas_weight: float = 0.7
    rbas_oversample: int = 3
    # GradNorm loss balancing
    use_gradnorm: bool = False
    gradnorm_alpha: float = 1.5


@dataclass
class TrainingSchedule:
    """Full multi-phase training schedule.

    Attributes:
        phases: List of training phases.
    """
    phases: List[TrainingPhase] = field(default_factory=list)

    def add_phase(
        self,
        name: str,
        steps: int,
        lr: float,
        **kwargs
    ) -> "TrainingSchedule":
        """Add a training phase.

        Args:
            name: Phase name.
            steps: Number of steps.
            lr: Learning rate.
            **kwargs: Additional phase configuration.

        Returns:
            Self for chaining.
        """
        self.phases.append(TrainingPhase(name=name, steps=steps, lr=lr, **kwargs))
        return self

    @classmethod
    def default_pinn(cls, total_steps: int = 2000) -> "TrainingSchedule":
        """Default 3-phase PINN training schedule.

        Args:
            total_steps: Total training steps (distributed across phases).

        Returns:
            Configured TrainingSchedule.
        """
        return cls(phases=[
            TrainingPhase(
                name="Phase 1: Initial",
                steps=int(total_steps * 0.5),
                lr=0.01,
                loss_weights={"PDEResidualLoss": 1.0, "BoundaryConditionLoss": 10.0},
                log_freq=max(1, total_steps // 20),
            ),
            TrainingPhase(
                name="Phase 2: Refinement",
                steps=int(total_steps * 0.3),
                lr=0.001,
                loss_weights={"PDEResidualLoss": 10.0, "SmoothnessLoss": 0.5},
                grid_update_before=True,
                log_freq=max(1, total_steps // 20),
            ),
            TrainingPhase(
                name="Phase 3: Fine-tune",
                steps=int(total_steps * 0.2),
                lr=0.0001,
                symbolic_after=True,
                log_freq=max(1, total_steps // 20),
            ),
        ])

    @classmethod
    def eigenvalue(cls, total_steps: int = 4000) -> "TrainingSchedule":
        """Training schedule for eigenvalue problems.

        Args:
            total_steps: Total training steps.

        Returns:
            Configured TrainingSchedule.
        """
        return cls(phases=[
            TrainingPhase(
                name="Phase 1: Shape",
                steps=int(total_steps * 0.5),
                lr=0.01,
                loss_weights={
                    "PDEResidualLoss": 10.0,
                    "NormalizationLoss": 10.0,
                    "NonTrivialLoss": 20.0,
                },
                log_freq=max(1, total_steps // 20),
            ),
            TrainingPhase(
                name="Phase 2: Energy",
                steps=int(total_steps * 0.3),
                lr=0.001,
                loss_weights={
                    "PDEResidualLoss": 50.0,
                    "EigenvalueLoss": 1.0,
                },
                grid_update_before=True,
                log_freq=max(1, total_steps // 20),
            ),
            TrainingPhase(
                name="Phase 3: Polish",
                steps=int(total_steps * 0.2),
                lr=0.0001,
                symbolic_after=True,
                log_freq=max(1, total_steps // 20),
            ),
        ])

    @classmethod
    def wigner(cls, total_steps: int = 4000) -> "TrainingSchedule":
        """Training schedule for Wigner function problems.

        Args:
            total_steps: Total training steps.

        Returns:
            Configured TrainingSchedule.
        """
        return cls(phases=[
            TrainingPhase(
                name="Phase 1: Initial",
                steps=int(total_steps * 0.6),
                lr=0.001,
                loss_weights={
                    "PDEResidualLoss": 10.0,
                    "NormalizationLoss": 10.0,
                    "NonTrivialLoss": 20.0,
                    "DecayLoss": 10.0,
                },
                log_freq=max(1, total_steps // 15),
            ),
            TrainingPhase(
                name="Phase 2: Refine",
                steps=int(total_steps * 0.4),
                lr=0.0003,
                loss_weights={
                    "PDEResidualLoss": 50.0,
                    "EigenvalueLoss": 1.0,
                },
                grid_update_before=True,
                log_freq=max(1, total_steps // 15),
            ),
        ])

    @classmethod
    def quick(cls, steps: int = 500) -> "TrainingSchedule":
        """Quick single-phase training for testing.

        Args:
            steps: Number of training steps.

        Returns:
            Simple TrainingSchedule.
        """
        return cls(phases=[
            TrainingPhase(
                name="Quick Training",
                steps=steps,
                lr=0.01,
                log_freq=max(1, steps // 10),
            ),
        ])

    @classmethod
    def lbfgs(cls, max_iter: int = 100, adam_warmup: int = 500) -> "TrainingSchedule":
        """L-BFGS training schedule with optional Adam warmup.

        L-BFGS often converges faster for physics problems but benefits
        from a short Adam warmup to get into a good region.

        Args:
            max_iter: Maximum L-BFGS iterations.
            adam_warmup: Adam warmup steps (0 to skip).

        Returns:
            Configured TrainingSchedule.
        """
        phases = []

        if adam_warmup > 0:
            phases.append(TrainingPhase(
                name="Adam Warmup",
                steps=adam_warmup,
                lr=0.01,
                log_freq=max(1, adam_warmup // 10),
            ))

        # L-BFGS phase uses negative steps to indicate L-BFGS mode
        # The trainer will detect this and switch to L-BFGS
        phases.append(TrainingPhase(
            name="L-BFGS Optimization",
            steps=-max_iter,  # Negative = L-BFGS iterations
            lr=1.0,  # Not used by L-BFGS
            log_freq=10,
        ))

        return cls(phases=phases)


# =============================================================================
# TRAINING HISTORY
# =============================================================================

@dataclass
class PhaseHistory:
    """Training history for one phase.

    Attributes:
        name: Phase name.
        steps: Number of steps completed.
        losses: List of total loss values.
        breakdowns: List of per-term loss breakdowns.
        eigenvalues: List of computed eigenvalues (if applicable).
        times: List of step times.
    """
    name: str
    steps: int = 0
    losses: List[float] = field(default_factory=list)
    breakdowns: List[Dict[str, float]] = field(default_factory=list)
    eigenvalues: List[float] = field(default_factory=list)
    times: List[float] = field(default_factory=list)


@dataclass
class TrainingHistory:
    """Complete training history.

    Attributes:
        phases: List of phase histories.
        total_time: Total training time in seconds.
        trainable_params: Final values of trainable parameters.

    REPL-friendly access:
        >>> h.loss      # Final loss
        >>> h.E         # Trained eigenvalue
        >>> h.eigenvalue  # Same as h.E
        >>> h           # Nice repr
    """
    phases: List[PhaseHistory] = field(default_factory=list)
    total_time: float = 0.0
    trainable_params: Dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        """REPL-friendly representation."""
        parts = ["TrainingHistory("]
        parts.append(f"  loss={self.final_loss:.6f},")
        if self.final_eigenvalue is not None:
            parts.append(f"  E={self.final_eigenvalue:.6f},")
        if self.trainable_params:
            params_str = ", ".join(f"{k}={v:.4f}" for k, v in self.trainable_params.items())
            parts.append(f"  trainable_params={{{params_str}}},")
        parts.append(f"  phases={len(self.phases)},")
        parts.append(f"  time={self.total_time:.2f}s")
        parts.append(")")
        return "\n".join(parts)

    @property
    def loss(self) -> float:
        """Shortcut for final_loss (REPL-friendly)."""
        return self.final_loss

    @property
    def E(self) -> Optional[float]:
        """Shortcut for eigenvalue (REPL-friendly)."""
        return self.final_eigenvalue

    @property
    def final_loss(self) -> float:
        """Get final loss value."""
        if self.phases and self.phases[-1].losses:
            return self.phases[-1].losses[-1]
        return float('inf')

    @property
    def final_eigenvalue(self) -> Optional[float]:
        """Get final eigenvalue (if computed or trained)."""
        # First check trainable params for E
        if "E" in self.trainable_params:
            return self.trainable_params["E"]
        # Fall back to computed eigenvalue
        if self.phases:
            for phase in reversed(self.phases):
                if phase.eigenvalues:
                    return phase.eigenvalues[-1]
        return None

    @property
    def eigenvalue(self) -> Optional[float]:
        """Alias for final_eigenvalue."""
        return self.final_eigenvalue

    def plot(self, show: bool = True, save: Optional[str] = None):
        """Plot training history (loss and eigenvalue curves).

        REPL-friendly: just call h.plot() to see training progress.

        Args:
            show: Whether to display the plot (default True).
            save: Optional path to save the figure.

        Returns:
            matplotlib Figure if available, None otherwise.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed. Run: pip install matplotlib")
            return None

        # Collect all losses across phases
        all_losses = []
        all_eigenvalues = []
        phase_boundaries = [0]

        for phase in self.phases:
            all_losses.extend(phase.losses)
            all_eigenvalues.extend(phase.eigenvalues)
            phase_boundaries.append(len(all_losses))

        if not all_losses:
            print("No training data to plot")
            return None

        # Create figure
        has_eigenvalues = len(all_eigenvalues) > 0
        fig, axes = plt.subplots(1, 2 if has_eigenvalues else 1, figsize=(10 if has_eigenvalues else 5, 4))

        if not has_eigenvalues:
            axes = [axes]

        # Plot loss
        ax = axes[0]
        ax.plot(all_losses, 'b-', linewidth=1.5)
        ax.set_xlabel('Log Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Mark phase boundaries
        for i, boundary in enumerate(phase_boundaries[1:-1], 1):
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)

        # Plot eigenvalue if available
        if has_eigenvalues:
            ax = axes[1]
            ax.plot(all_eigenvalues, 'r-', linewidth=1.5)
            ax.set_xlabel('Log Step')
            ax.set_ylabel('E')
            ax.set_title(f'Eigenvalue (final: {self.E:.4f})' if self.E else 'Eigenvalue')
            ax.grid(True, alpha=0.3)

            # Mark phase boundaries
            for boundary in phase_boundaries[1:-1]:
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()

        if save:
            plt.savefig(save, dpi=150, bbox_inches='tight')
            print(f"Saved to {save}")

        if show:
            plt.show()

        return fig

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'phases': [
                {
                    'name': p.name,
                    'steps': p.steps,
                    'losses': p.losses,
                    'breakdowns': p.breakdowns,
                    'eigenvalues': p.eigenvalues,
                    'times': p.times,
                }
                for p in self.phases
            ],
            'total_time': self.total_time,
            'final_loss': self.final_loss,
            'final_eigenvalue': self.final_eigenvalue,
            'trainable_params': self.trainable_params,
        }


# =============================================================================
# PDE TRAINER
# =============================================================================

class PDETrainer:
    """Multi-phase trainer for PDE problems.

    Orchestrates the training process including:
    - Multi-phase training with different configurations
    - Loss composition and weighting
    - Grid refinement
    - Symbolic extraction
    - Progress logging

    Example:
        >>> from kan_mlx_physics.pde import PDETrainer, TrainingSchedule

        >>> schedule = TrainingSchedule.eigenvalue(total_steps=3000)
        >>> trainer = PDETrainer(problem, model, schedule)
        >>> history = trainer.train(verbose=True)

        >>> print(f"Final loss: {history.final_loss:.6f}")
        >>> print(f"Eigenvalue: {history.final_eigenvalue:.4f}")
    """

    def __init__(
        self,
        model: Any,
        domain: Domain,
        compiled_residual: CompiledResidual,
        loss_composer: LossComposer,
        schedule: TrainingSchedule,
        params: Dict[str, float] = None,
        trainable_param_names: Optional[Set[str]] = None,
        pinn_trainer: Optional[Any] = None,
    ):
        """Initialize the PDE trainer.

        Args:
            model: MultKAN model to train.
            domain: Problem domain for sampling.
            compiled_residual: Compiled PDE residual.
            loss_composer: Loss term composer.
            schedule: Training schedule.
            params: All parameter values (fixed and initial values for trainable).
            trainable_param_names: Set of parameter names to train via Adam.
            pinn_trainer: Optional PINNTrainer for derivatives.
        """
        self.model = model
        self.domain = domain
        self.compiled_residual = compiled_residual
        self.loss_composer = loss_composer
        self.schedule = schedule
        self.pinn_trainer = pinn_trainer

        # Separate fixed params from trainable params
        all_params = params or {}
        self.trainable_param_names = trainable_param_names or set()
        self.trainable_params: Dict[str, mx.array] = {}
        self.fixed_params: Dict[str, float] = {}

        for name, value in all_params.items():
            if name in self.trainable_param_names:
                self.trainable_params[name] = mx.array([float(value)])
            else:
                self.fixed_params[name] = value

        # Adam state for trainable params
        self.m_trainable: Dict[str, mx.array] = {
            k: mx.zeros_like(v) for k, v in self.trainable_params.items()
        }
        self.v_trainable: Dict[str, mx.array] = {
            k: mx.zeros_like(v) for k, v in self.trainable_params.items()
        }
        self.trainable_step = 0  # Track step for Adam bias correction

        # Adam state for model params (initialized lazily when trainable params exist)
        self.m_model = None
        self.v_model = None

        # For backwards compatibility
        self.params = self.fixed_params

        # Create PINNTrainer if not provided
        if self.pinn_trainer is None:
            try:
                from ..pinn import PINNTrainer
                self.pinn_trainer = PINNTrainer(
                    model,
                    lr=schedule.phases[0].lr if schedule.phases else 0.01,
                    derivative_method="finite_diff"
                )
            except ImportError:
                pass

        self.history = TrainingHistory()
        self._live_plotter = None  # Injected externally by pde() if live_plot() was called

    def train(self, verbose: bool = True) -> TrainingHistory:
        """Execute the full training schedule.

        Args:
            verbose: Whether to print progress.

        Returns:
            TrainingHistory with all logged data.
        """
        start_time = time.time()

        for phase in self.schedule.phases:
            if verbose:
                print(f"\n{'='*60}")
                print(f"{phase.name}")
                print(f"{'='*60}")

            # Attach live plotter on the first phase (lazy, so model is ready)
            if self._live_plotter is not None and not getattr(self._live_plotter, '_attached', False):
                self._live_plotter.post_attach(self.model)
                self._live_plotter._attached = True

            # Grid update before phase
            if phase.grid_update_before and not phase.disable_grid_updates:
                self._update_grid(verbose)

            # Prune before phase
            if phase.prune_before:
                self._prune_model(phase.prune_threshold, verbose)

            # Apply phase configuration
            self._apply_phase_config(phase)

            # Set learning rate
            if self.pinn_trainer is not None:
                self.pinn_trainer.lr = phase.lr

            # Train this phase
            phase_history = self._train_phase(phase, verbose)
            self.history.phases.append(phase_history)

            # Grid update after phase
            if phase.grid_update_after and not phase.disable_grid_updates:
                self._update_grid(verbose)

            # Prune after phase
            if phase.prune_after:
                self._prune_model(phase.prune_threshold, verbose)

            # Symbolic extraction
            if phase.symbolic_after:
                self._extract_symbolic(phase.symbolic_threshold, verbose)

            # Live plotter: notify phase end
            if self._live_plotter is not None:
                self._live_plotter.on_phase_end(self.model, phase.name)

        self.history.total_time = time.time() - start_time

        # Store final trainable param values in history
        for name, value in self.trainable_params.items():
            self.history.trainable_params[name] = float(value[0]) if value.shape else float(value)

        if verbose:
            print(f"\n{'='*60}")
            print("Training Complete")
            print(f"{'='*60}")
            print(f"Total time: {self.history.total_time:.2f}s")
            print(f"Final loss: {self.history.final_loss:.6f}")
            if self.history.final_eigenvalue is not None:
                print(f"Eigenvalue: {self.history.final_eigenvalue:.6f}")

        return self.history

    def _apply_phase_config(self, phase: TrainingPhase):
        """Apply phase-specific loss configuration."""
        # Update weights
        for term_id, weight in phase.loss_weights.items():
            self.loss_composer.set_weight(term_id, weight)

        # Enable/disable terms
        if phase.enabled_losses is not None:
            self.loss_composer.enable_only(phase.enabled_losses)

        if phase.disabled_losses is not None:
            for term_id in phase.disabled_losses:
                self.loss_composer.disable(term_id)

    def _train_phase(self, phase: TrainingPhase, verbose: bool) -> PhaseHistory:
        """Train for one phase.

        Args:
            phase: Phase configuration.
            verbose: Whether to print progress.

        Returns:
            PhaseHistory for this phase.
        """
        history = PhaseHistory(name=phase.name)

        # Check for L-BFGS mode (negative steps)
        if phase.steps < 0:
            return self._train_lbfgs_phase(phase, -phase.steps, verbose)

        # Initial sampling
        x_interior = self._sample_interior(phase.n_points)
        x_boundary = self._sample_boundary(phase.n_boundary)

        for step in range(phase.steps):
            step_start = time.time()

            # Resample periodically
            if step > 0 and step % phase.resample_freq == 0:
                x_interior = self._sample_interior(phase.n_points)
                x_boundary = self._sample_boundary(phase.n_boundary)

            # Build loss context with both fixed and trainable params
            ctx = LossContext(
                model=self.model,
                trainer=self.pinn_trainer,
                x_interior=x_interior,
                x_boundary=x_boundary,
                params=self.fixed_params,
                trainable_params=self.trainable_params,
                compiled_residual=self.compiled_residual,
                domain=self.domain,
            )

            # Training step (handles both model and trainable param gradients)
            loss = self._training_step(ctx, phase.lr)
            mx.eval(loss)  # CRITICAL: Prevent graph buildup

            step_time = time.time() - step_start

            # Logging
            if step % phase.log_freq == 0 or step == phase.steps - 1:
                # Compute detailed breakdown
                _, breakdown = self.loss_composer.compute_total(ctx)

                history.losses.append(float(loss))
                history.breakdowns.append(breakdown)
                history.times.append(step_time)

                # Record eigenvalue from trainable params or computed
                if "E" in self.trainable_params:
                    E_val = self.trainable_params["E"]
                    history.eigenvalues.append(float(E_val[0]) if E_val.shape else float(E_val))
                elif ctx.computed_eigenvalue is not None:
                    history.eigenvalues.append(float(ctx.computed_eigenvalue))

                if verbose:
                    self._log_step(step, phase.steps, loss, breakdown, ctx)

                # Live plot update
                if self._live_plotter is not None:
                    self._live_plotter.on_step(step, float(loss))

            history.steps = step + 1

        return history

    def _train_lbfgs_phase(
        self,
        phase: TrainingPhase,
        max_iter: int,
        verbose: bool
    ) -> PhaseHistory:
        """Train using L-BFGS optimizer.

        Args:
            phase: Phase configuration.
            max_iter: Maximum L-BFGS iterations.
            verbose: Whether to print progress.

        Returns:
            PhaseHistory for this phase.
        """
        history = PhaseHistory(name=phase.name)

        try:
            from ..functional import (
                lbfgs_optimize,
                get_params_list,
                set_params_list,
                functional_forward,
            )
        except ImportError:
            if verbose:
                print("L-BFGS not available. Falling back to Adam.")
            # Fall back to regular training
            phase.steps = max_iter * 10  # Rough equivalent
            return self._train_phase(phase, verbose)

        # Sample points for L-BFGS (fixed during optimization)
        x_interior = self._sample_interior(phase.n_points)
        x_boundary = self._sample_boundary(phase.n_boundary)

        # Get model parameters
        params_list = get_params_list(self.model)
        k = self.model.k
        base_fun = self.model.base_fun

        # Build loss function for L-BFGS
        def lbfgs_loss_fn(params):
            # Temporarily update model for loss computation
            set_params_list(self.model, params)

            # Build context
            ctx = LossContext(
                model=self.model,
                trainer=self.pinn_trainer,
                x_interior=x_interior,
                x_boundary=x_boundary,
                params=self.params,
                compiled_residual=self.compiled_residual,
                domain=self.domain,
            )

            total, _ = self.loss_composer.compute_total(ctx, breakdown=False)
            return total

        # Track iterations for logging
        iteration_count = [0]
        start_time = time.time()

        def callback(xk, iteration):
            iteration_count[0] = iteration
            if verbose and iteration % phase.log_freq == 0:
                # Compute current loss for logging
                loss = lbfgs_loss_fn(get_params_list(self.model))
                print(f"  L-BFGS iter {iteration}: loss = {float(loss):.6f}")
                history.losses.append(float(loss))

        if verbose:
            print(f"Starting L-BFGS optimization (max_iter={max_iter})")

        # Run L-BFGS
        final_params, result = lbfgs_optimize(
            lbfgs_loss_fn,
            params_list,
            max_iter=max_iter,
            callback=callback,
            verbose=False,  # We handle our own logging
        )

        # Update model with final parameters
        set_params_list(self.model, final_params)

        history.steps = iteration_count[0]
        history.times.append(time.time() - start_time)

        if verbose:
            print(f"L-BFGS finished: {result.message}")
            print(f"  Final loss: {result.fun:.6f}")
            print(f"  Iterations: {result.nit}")

        return history

    def _training_step(self, ctx: LossContext, lr: float) -> mx.array:
        """Execute one training step.

        Args:
            ctx: Loss context.
            lr: Learning rate for this step.

        Returns:
            Total loss value.
        """
        # If we have trainable params, we need custom gradient handling
        if self.trainable_params:
            return self._training_step_with_trainable_params(ctx, lr)

        # Standard training (no trainable params)
        if self.pinn_trainer is not None:
            # Use PINNTrainer's step method
            def loss_fn(x):
                ctx.x_interior = x
                total, _ = self.loss_composer.compute_total(ctx, breakdown=False)
                return total

            return self.pinn_trainer.step(loss_fn, ctx.x_interior)

        else:
            # Manual gradient descent
            def loss_fn():
                total, _ = self.loss_composer.compute_total(ctx, breakdown=False)
                return total

            loss, grads = mx.value_and_grad(loss_fn)()
            # Simple SGD update (would need proper optimizer)
            mx.eval(loss)
            return loss

    def _is_bspline_model(self) -> bool:
        """Check if all layers use B-spline basis (functional_forward compatible)."""
        for layer in self.model.layers:
            if getattr(layer, 'basis_type', 'bspline') != 'bspline':
                return False
        return True

    def _training_step_with_trainable_params(
        self, ctx: LossContext, lr: float
    ) -> mx.array:
        """Training step with trainable parameters.

        Always uses the model-based path which works with any basis type
        and uses the loss composer (not hardcoded PDE).
        """
        return self._training_step_trainable_model_based(ctx, lr)

    def _training_step_trainable_model_based(
        self, ctx: LossContext, lr: float
    ) -> mx.array:
        """Training step for non-B-spline bases (Laguerre, Hermite, etc.).

        Uses model(x) for forward passes (works with any basis) and the loss
        composer for computing the total loss. Gradients for model params and
        trainable params are computed together in a single fused
        mx.value_and_grad pass (argnums=(0,1)).
        """
        self.trainable_step += 1
        t = mx.array(self.trainable_step, dtype=mx.float32)

        model = self.model

        def make_loss(model_params, tp):
            # Bind the differentiated model-parameter tree onto the model so the
            # forward pass inside compute_total sees it.
            model.update(model_params)
            local_ctx = LossContext(
                model=model,
                trainer=self.pinn_trainer,
                x_interior=ctx.x_interior,
                x_boundary=ctx.x_boundary,
                params=ctx.params,
                trainable_params=tp,
                compiled_residual=ctx.compiled_residual,
                domain=ctx.domain,
                boundary_conditions=ctx.boundary_conditions,
            )
            total, _ = self.loss_composer.compute_total(local_ctx, breakdown=False)
            return total

        # Fused single forward/backward: model-param grads AND trainable-param
        # grads in one pass (argnums=(0,1)), instead of differentiating the full
        # second-order loss twice. Roughly halves the work on the non-B-spline
        # (Laguerre/Hermite) path — see benchmark_kan_vs_pykan.py.
        loss_and_grad = mx.value_and_grad(make_loss, argnums=(0, 1))
        loss, (model_grads, g_trainable) = loss_and_grad(
            model.trainable_parameters(), self.trainable_params
        )
        mx.eval(loss, model_grads, *g_trainable.values())

        # Update model params with Adam
        if not hasattr(self, '_mlx_optimizer_model'):
            import mlx.optimizers as optim
            self._mlx_optimizer_model = optim.Adam(learning_rate=lr)
        else:
            self._mlx_optimizer_model.learning_rate = lr

        self._mlx_optimizer_model.update(model, model_grads)
        mx.eval(model.parameters())

        # Adam for trainable params
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, param in list(self.trainable_params.items()):
            g = g_trainable[name]
            self.m_trainable[name] = beta1 * self.m_trainable[name] + (1 - beta1) * g
            self.v_trainable[name] = beta2 * self.v_trainable[name] + (1 - beta2) * (g ** 2)
            m_hat = self.m_trainable[name] / (1 - beta1 ** t)
            v_hat = self.v_trainable[name] / (1 - beta2 ** t)
            self.trainable_params[name] = param - lr * m_hat / (mx.sqrt(v_hat) + eps)
            self.trainable_params[name] = mx.maximum(self.trainable_params[name], mx.array([0.01]))

        mx.eval(*self.trainable_params.values())
        return loss

    def _training_step_trainable_functional(
        self, ctx: LossContext, lr: float
    ) -> mx.array:
        """Training step using B-spline functional API for single forward/backward pass.

        Matches the manual implementation exactly: computes gradients for
        both model parameters and trainable parameters (E) in one pass.
        Only works with B-spline basis.
        """
        from ..functional import get_params_list, set_params_list, adam_update, init_adam_state
        from ..pinn import make_derivative_fns

        self.trainable_step += 1
        t = mx.array(self.trainable_step, dtype=mx.float32)

        # Get functional components (cached after first call)
        if not hasattr(self, '_u_fn'):
            self._u_fn, self._du_fn, self._d2u_fn = make_derivative_fns(
                self.model.k, self.model.base_fun
            )

        # Get model params and init Adam state if needed
        params_list = get_params_list(self.model)
        if self.m_model is None:
            self.m_model, self.v_model = init_adam_state(params_list)

        # Combined loss for single backward pass
        def loss_fn(params_list, trainable_dict):
            x, x_bc = ctx.x_interior, ctx.x_boundary
            u = self._u_fn(params_list, x)
            d2u = self._d2u_fn(params_list, x)

            # Merge params
            E = trainable_dict.get("E", mx.array([1.0]))
            e = E[0] if E.shape else E

            # PDE: 0.5 * d²u/dx² + E * u = 0
            pde_residual = 0.5 * d2u + e * u
            pde_loss = mx.mean(pde_residual ** 2) / (e ** 2 + 1e-6)

            # Boundary: u(0) = u(L) = 0
            u_bc = self._u_fn(params_list, x_bc) if x_bc is not None and x_bc.shape[0] > 0 else None
            bc_loss = mx.mean(u_bc ** 2) if u_bc is not None else mx.array(0.0)

            # Get domain volume for integrals
            L = 1.0
            if ctx.domain and hasattr(ctx.domain, 'bounds'):
                for lo, hi in ctx.domain.bounds:
                    L *= (hi - lo)

            # Normalization: ∫|u|² dx ≈ 1
            integral_u = mx.mean(u ** 2) * L
            norm_loss = (integral_u - 1.0) ** 2

            # Non-trivial: exp(-10 * ∫|u|²)
            nontrivial_loss = mx.exp(-10.0 * integral_u)

            # Eigenvalue positivity: exp(-10|E|)
            eig_loss = mx.exp(-10.0 * mx.abs(e))

            # Get weights from loss terms
            w = {t.__class__.__name__: t.weight for t in self.loss_composer.terms if t.enabled}

            return (
                w.get("PDEResidualLoss", 500) * pde_loss +
                w.get("BoundaryConditionLoss", 1000) * bc_loss +
                w.get("NormalizationLoss", 100) * norm_loss +
                w.get("NonTrivialLoss", 200) * nontrivial_loss +
                w.get("EigenvalueLoss", 3) * eig_loss
            )

        # Single backward pass for both params
        loss, (g_params, g_trainable) = mx.value_and_grad(loss_fn, argnums=(0, 1))(
            params_list, self.trainable_params
        )
        mx.eval(loss, g_params, *g_trainable.values())  # CRITICAL: Eval grads immediately

        # Adam updates
        params_list, self.m_model, self.v_model = adam_update(
            params_list, g_params, self.m_model, self.v_model,
            t, lr=lr, beta1=0.9, beta2=0.999, eps=1e-8
        )
        mx.eval(params_list, self.m_model, self.v_model)  # CRITICAL: Eval after Adam update

        # Update model with new params (unpack tuples)
        for i, (coef, scale_sp, scale_base, grid) in enumerate(params_list):
            self.model.layers[i].coef = coef
            self.model.layers[i].scale_sp = scale_sp
            self.model.layers[i].scale_base = scale_base

        # Adam for trainable params
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for name, param in list(self.trainable_params.items()):
            g = g_trainable[name]
            self.m_trainable[name] = beta1 * self.m_trainable[name] + (1 - beta1) * g
            self.v_trainable[name] = beta2 * self.v_trainable[name] + (1 - beta2) * (g ** 2)
            m_hat = self.m_trainable[name] / (1 - beta1 ** t)
            v_hat = self.v_trainable[name] / (1 - beta2 ** t)
            self.trainable_params[name] = param - lr * m_hat / (mx.sqrt(v_hat) + eps)
            self.trainable_params[name] = mx.maximum(self.trainable_params[name], mx.array([0.01]))

        mx.eval(loss, params_list, self.m_model, self.v_model, *self.trainable_params.values())
        return loss

    def _sample_interior(
        self,
        n_points: int,
        use_rbas: bool = False,
        rbas_weight: float = 0.7,
        rbas_oversample: int = 3,
    ) -> mx.array:
        """Sample interior collocation points.

        Args:
            n_points: Number of points to sample.
            use_rbas: Use residual-based adaptive sampling.
            rbas_weight: Weight for RBAS vs uniform (0-1).
            rbas_oversample: Oversample factor for RBAS candidates.

        Returns:
            Sampled points of shape (n_points, dim).
        """
        if not use_rbas:
            # Standard uniform sampling
            if self.domain is not None:
                return self.domain.sample(n_points, strategy="uniform")
            else:
                return mx.random.uniform(shape=(n_points, 1))

        # RBAS: Residual-Based Adaptive Sampling
        return self._sample_interior_rbas(n_points, rbas_weight, rbas_oversample)

    def _sample_interior_rbas(
        self,
        n_points: int,
        rbas_weight: float = 0.7,
        rbas_oversample: int = 3,
    ) -> mx.array:
        """Sample interior points using residual-based adaptive sampling.

        Oversamples where |R(x)| is largest to focus training on
        difficult regions.

        Args:
            n_points: Target number of points.
            rbas_weight: Fraction of points from importance sampling (0-1).
            rbas_oversample: Factor to oversample candidate pool.

        Returns:
            Points concentrated in high-residual regions.
        """
        # 1. Sample candidate pool (larger than needed)
        n_candidates = n_points * rbas_oversample
        if self.domain is not None:
            candidates = self.domain.sample(n_candidates, strategy="uniform")
        else:
            candidates = mx.random.uniform(shape=(n_candidates, 1))

        # 2. Compute residuals WITHOUT gradients (for selection only)
        # We need the compiled residual and current params
        if self.compiled_residual is None or self.model is None:
            # Fall back to uniform if not set up yet
            return candidates[:n_points]

        try:
            # Evaluate residual at candidates
            residuals = self.compiled_residual.evaluate(
                self.model, candidates, self.fixed_params
            )
            residuals = mx.abs(residuals)
            mx.eval(residuals)

            # 3. Compute importance weights
            residuals_np = np.array(residuals).flatten()

            # Add small epsilon to avoid zero weights
            weights = residuals_np + 1e-8
            weights = weights / weights.sum()

            # 4. Split into uniform and importance-sampled portions
            n_importance = int(n_points * rbas_weight)
            n_uniform = n_points - n_importance

            # Uniform samples (random subset)
            uniform_idx = np.random.choice(n_candidates, size=n_uniform, replace=False)

            # Importance-weighted samples
            # Higher residual = higher probability of selection
            importance_idx = np.random.choice(
                n_candidates,
                size=n_importance,
                replace=False,
                p=weights,
            )

            # Combine indices
            all_idx = np.concatenate([uniform_idx, importance_idx])
            np.random.shuffle(all_idx)  # Shuffle to mix uniform and importance

            return candidates[mx.array(all_idx)]

        except Exception:
            # Fall back to uniform on any error
            return candidates[:n_points]

    def _sample_boundary(self, n_points: int) -> mx.array:
        """Sample boundary collocation points."""
        if self.domain is not None:
            # Use sample_boundary for actual boundary points
            if hasattr(self.domain, 'sample_boundary'):
                return self.domain.sample_boundary(n_points)
            # Fall back to boundary strategy
            try:
                return self.domain.sample(n_points, strategy="boundary")
            except (TypeError, ValueError):
                # Boundary strategy might not be supported - sample near edges
                interior = self.domain.sample(n_points, strategy="uniform")
                return interior  # Fall back to interior points
        else:
            # Default: endpoints of unit interval
            n_each = n_points // 2
            left = mx.zeros((n_each, 1))
            right = mx.ones((n_each, 1))
            return mx.concatenate([left, right], axis=0)

    def _update_grid(self, verbose: bool):
        """Update the model grid from samples."""
        if verbose:
            print("Updating grid...")

        if hasattr(self.model, 'update_grid_from_samples'):
            x_sample = self._sample_interior(1000)
            self.model.update_grid_from_samples(x_sample)

            if verbose:
                print("Grid updated")

    def _prune_model(self, threshold: float, verbose: bool):
        """Prune the model by removing low-importance edges.

        Args:
            threshold: Prune edges with importance below this threshold.
            verbose: Whether to print progress.
        """
        if verbose:
            print(f"Pruning model (threshold={threshold})...")

        if hasattr(self.model, 'prune'):
            try:
                # Get model structure before pruning
                before = str(self.model.width) if hasattr(self.model, 'width') else "unknown"

                # Prune the model
                self.model.prune(threshold=threshold)

                # Get model structure after pruning
                after = str(self.model.width) if hasattr(self.model, 'width') else "unknown"

                if verbose:
                    print(f"Pruned: {before} -> {after}")

                # Need to recreate PINNTrainer after pruning since model structure changed
                if self.pinn_trainer is not None:
                    try:
                        from ..pinn import PINNTrainer
                        old_lr = self.pinn_trainer.lr
                        self.pinn_trainer = PINNTrainer(
                            self.model,
                            lr=old_lr,
                            derivative_method="finite_diff"
                        )
                    except ImportError:
                        pass

            except Exception as e:
                if verbose:
                    print(f"Pruning failed: {e}")
        else:
            if verbose:
                print("Model does not support pruning")

    def _extract_symbolic(self, threshold: float, verbose: bool):
        """Extract symbolic formulas from the model.

        Args:
            threshold: R² threshold for auto_symbolic.
            verbose: Whether to print progress.
        """
        if verbose:
            print(f"Extracting symbolic formulas (R² threshold={threshold})...")

        if hasattr(self.model, 'auto_symbolic'):
            x_sample = self._sample_interior(1000)
            try:
                fixed_edges = self.model.auto_symbolic(
                    x=x_sample,
                    r2_threshold=threshold,
                    verbose=verbose,
                    basis_only=True,
                )

                if verbose:
                    if fixed_edges:
                        print(f"Fixed {len(fixed_edges)} edges to symbolic functions")

                    if hasattr(self.model, 'symbolic_formula'):
                        formula = self.model.symbolic_formula(decimals=3)
                        print(f"Extracted formula: {formula}")
            except Exception as e:
                if verbose:
                    print(f"Symbolic extraction failed: {e}")

    def _log_step(
        self,
        step: int,
        total_steps: int,
        loss: mx.array,
        breakdown: Dict[str, float],
        ctx: LossContext
    ):
        """Log training progress."""
        parts = [f"Step {step:4d}/{total_steps}"]
        parts.append(f"Loss: {float(loss):.4f}")

        # Add key metrics
        for name, val in breakdown.items():
            short_name = name.replace("Loss", "")[:8]
            parts.append(f"{short_name}: {val:.4f}")

        # Add trainable eigenvalue or computed eigenvalue
        if "E" in self.trainable_params:
            E_val = self.trainable_params["E"]
            parts.append(f"E: {float(E_val[0]) if E_val.shape else float(E_val):.4f}")
        elif ctx.computed_eigenvalue is not None:
            parts.append(f"E: {float(ctx.computed_eigenvalue):.4f}")

        print(" | ".join(parts))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_trainer(
    model: Any,
    domain: Domain,
    compiled_residual: CompiledResidual,
    losses: Optional[List[LossTerm]] = None,
    schedule: Optional[TrainingSchedule] = None,
    params: Optional[Dict[str, float]] = None,
) -> PDETrainer:
    """Create a PDETrainer with default configuration.

    Args:
        model: MultKAN model.
        domain: Problem domain.
        compiled_residual: Compiled residual.
        losses: List of loss terms (default: standard PINN losses).
        schedule: Training schedule (default: eigenvalue schedule).
        params: Parameter values.

    Returns:
        Configured PDETrainer.
    """
    # Build loss composer
    composer = LossComposer()
    if losses:
        for loss in losses:
            composer.add(loss)
    else:
        from .losses import default_pinn_losses
        composer = default_pinn_losses()

    # Default schedule
    if schedule is None:
        schedule = TrainingSchedule.eigenvalue()

    return PDETrainer(
        model=model,
        domain=domain,
        compiled_residual=compiled_residual,
        loss_composer=composer,
        schedule=schedule,
        params=params or {},
    )


def train_pde(
    model: Any,
    domain: Domain,
    compiled_residual: CompiledResidual,
    losses: Optional[List[LossTerm]] = None,
    schedule: Optional[TrainingSchedule] = None,
    params: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> TrainingHistory:
    """Train a PDE model with default settings.

    This is a convenience function that creates a trainer and runs it.

    Args:
        model: MultKAN model.
        domain: Problem domain.
        compiled_residual: Compiled residual.
        losses: List of loss terms.
        schedule: Training schedule.
        params: Parameter values.
        verbose: Whether to print progress.

    Returns:
        TrainingHistory.

    Example:
        >>> history = train_pde(model, domain, compiled, verbose=True)
        >>> print(f"Final loss: {history.final_loss:.6f}")
    """
    trainer = create_trainer(
        model=model,
        domain=domain,
        compiled_residual=compiled_residual,
        losses=losses,
        schedule=schedule,
        params=params,
    )
    return trainer.train(verbose=verbose)
