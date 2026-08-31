"""Modular Loss System for PDE Solving.

Provides composable loss terms that can be combined for different
PDE problems. Each loss term is independent and can be enabled/disabled
or have its weight adjusted during training.

Built-in loss terms:
- PDEResidualLoss: Main PDE constraint ||R[u]||²
- BoundaryConditionLoss: Enforce u=f on boundary
- NormalizationLoss: ∫|u|² = 1 for wavefunctions
- NonTrivialLoss: Prevent trivial u=0 solution
- EigenvalueLoss: Minimize eigenvalue (Rayleigh or trainable)
- SmoothnessLoss: Penalize large derivatives
- DecayLoss: u → 0 at boundaries
- AnchorLoss: u(x₀) = target
- RegularizationLoss: L1/L2 on model parameters

Example:
    >>> composer = LossComposer()
    >>> composer.add(PDEResidualLoss(weight=1.0))
    >>> composer.add(BoundaryConditionLoss(weight=10.0))
    >>> composer.add(NormalizationLoss(weight=5.0))

    >>> total_loss, breakdown = composer.compute_total(ctx)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple, Set, Union
import numpy as np

import mlx.core as mx

from .compiler import CompiledResidual
from .problem import Domain, BoundaryCondition


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_domain_volume(domain: Optional[Domain]) -> float:
    """Compute volume of a domain from its bounds."""
    if domain is None:
        return 1.0

    # Try to get volume attribute first
    if hasattr(domain, 'volume'):
        return domain.volume

    # Compute from bounds
    if hasattr(domain, 'bounds'):
        volume = 1.0
        for low, high in domain.bounds:
            volume *= (high - low)
        return volume

    return 1.0


# =============================================================================
# LOSS CONTEXT
# =============================================================================

@dataclass
class LossContext:
    """All data needed for loss computation.

    This context is passed to each loss term, providing access to
    the model, trainer, points, parameters, and compiled residual.

    Attributes:
        model: The MultKAN model.
        trainer: PINNTrainer instance (optional, for derivative access).
        x_interior: Interior collocation points.
        x_boundary: Boundary collocation points.
        params: Fixed parameter values.
        trainable_params: Trainable parameters (like eigenvalue E).
        compiled_residual: Compiled PDE residual.
        domain: Problem domain.
        boundary_conditions: List of boundary conditions.
        computed_eigenvalue: Eigenvalue computed by EigenvalueLoss (shared).
    """
    model: Any
    trainer: Optional[Any] = None
    x_interior: Optional[mx.array] = None
    x_boundary: Optional[mx.array] = None
    params: Dict[str, float] = field(default_factory=dict)
    trainable_params: Dict[str, mx.array] = field(default_factory=dict)
    compiled_residual: Optional[CompiledResidual] = None
    domain: Optional[Domain] = None
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    computed_eigenvalue: Optional[mx.array] = None

    def get_u(self, x: mx.array) -> mx.array:
        """Get function value at points x."""
        if self.trainer is not None:
            return self.trainer.u(x)
        return self.model(x)

    def get_du(self, x: mx.array) -> mx.array:
        """Get first derivative at points x."""
        if self.trainer is not None:
            return self.trainer.du(x)
        # Fallback: finite differences
        h = 1e-4
        return (self.model(x + h) - self.model(x - h)) / (2 * h)

    def get_d2u(self, x: mx.array) -> mx.array:
        """Get second derivative at points x."""
        if self.trainer is not None:
            return self.trainer.d2u(x)
        # Fallback: finite differences
        h = 1e-4
        return (self.model(x + h) - 2 * self.model(x) + self.model(x - h)) / (h ** 2)


# =============================================================================
# BASE LOSS TERM
# =============================================================================

class LossTerm(ABC):
    """Abstract base class for loss terms.

    Each loss term computes a scalar loss value from the LossContext.
    Loss terms can be enabled/disabled and have adjustable weights.

    Attributes:
        id: Unique identifier for this loss term.
        weight: Multiplicative weight for this loss.
        enabled: Whether this loss is active.
    """

    def __init__(self, weight: float = 1.0, enabled: bool = True, id: Optional[str] = None):
        self.id = id or self.__class__.__name__
        self.weight = weight
        self.enabled = enabled

    @abstractmethod
    def compute(self, ctx: LossContext) -> mx.array:
        """Compute this loss term.

        Args:
            ctx: Loss context with all required data.

        Returns:
            Scalar loss value.
        """
        pass

    def __repr__(self):
        status = "enabled" if self.enabled else "disabled"
        return f"{self.id}(weight={self.weight}, {status})"


# =============================================================================
# BUILT-IN LOSS TERMS
# =============================================================================

class PDEResidualLoss(LossTerm):
    """PDE residual loss: ||R[u](x)||².

    The main physics constraint - the PDE residual should be zero
    at all interior points.

    Args:
        weight: Loss weight (default: 1.0).
        normalize: Whether to normalize by u² norm (default: True).
        reduction: How to reduce over points ('mean' or 'sum').
    """

    def __init__(
        self,
        weight: float = 1.0,
        normalize: bool = True,
        reduction: str = 'mean',
        **kwargs
    ):
        super().__init__(weight, **kwargs)
        self.normalize = normalize
        self.reduction = reduction

    def compute(self, ctx: LossContext) -> mx.array:
        if ctx.compiled_residual is None or ctx.x_interior is None:
            return mx.array(0.0)

        # Merge fixed params with trainable params (keep mx.arrays for gradient flow)
        all_params = dict(ctx.params)  # Start with fixed params
        for name, value in ctx.trainable_params.items():
            # Keep trainable params as mx.array to preserve gradient flow
            all_params[name] = value

        # Compute residual (eigenvalue passed through params now)
        residual = ctx.compiled_residual.evaluate(
            ctx.trainer or ctx.model,
            ctx.x_interior,
            all_params,
            eigenvalue=None  # Eigenvalue is in params now
        )

        # Compute loss
        if self.reduction == 'mean':
            loss = mx.mean(residual ** 2)
        else:
            loss = mx.sum(residual ** 2)

        # Normalize by E² for trainable eigenvalue problems, but stop gradient
        # through the normalization factor to avoid biasing E upward.
        if "E" in ctx.trainable_params:
            E = ctx.trainable_params["E"]
            E_val = E[0] if E.shape else E
            # stop_gradient prevents E² normalization from driving E upward
            loss = loss / (mx.stop_gradient(E_val) ** 2 + 1e-6)

        # normalize=True divides by <u²>, making the residual scale-relative
        # and domain-size invariant. Applied independently of the E branch.
        if self.normalize:
            u = ctx.get_u(ctx.x_interior)
            if len(u.shape) > 1:
                u = mx.squeeze(u, axis=-1)
            u_norm = mx.mean(u ** 2) + 1e-6
            loss = loss / u_norm

        return loss


class BoundaryConditionLoss(LossTerm):
    """Boundary condition loss.

    Enforces boundary conditions by penalizing deviation from
    specified values on the boundary.

    Supports:
    - Dirichlet: u = f on boundary
    - Neumann: du/dn = g on boundary (simplified)
    - String specification: "dirichlet", "neumann", "periodic"

    Args:
        weight: Loss weight (default: 10.0 - BCs are usually important).
        bc_type: Type of BC ("dirichlet", "neumann", "periodic").
        target: Target value (for Dirichlet) or callable.
    """

    def __init__(
        self,
        weight: float = 10.0,
        bc_type: str = "dirichlet",
        target: Union[float, Callable] = 0.0,
        **kwargs
    ):
        super().__init__(weight, **kwargs)
        self.bc_type = bc_type.lower()
        self.target = target

    def compute(self, ctx: LossContext) -> mx.array:
        if ctx.x_boundary is None or ctx.x_boundary.shape[0] == 0:
            return mx.array(0.0)

        u_bc = ctx.get_u(ctx.x_boundary)
        if len(u_bc.shape) > 1:
            u_bc = mx.squeeze(u_bc, axis=-1)

        if self.bc_type == "dirichlet":
            # u = target on boundary
            if callable(self.target):
                target_vals = self.target(ctx.x_boundary)
            else:
                target_vals = mx.full((ctx.x_boundary.shape[0],), self.target)

            return mx.mean((u_bc - target_vals) ** 2)

        elif self.bc_type == "neumann":
            # du/dn = target on boundary (simplified: du/dx = target)
            du_bc = ctx.get_du(ctx.x_boundary)
            if len(du_bc.shape) > 1:
                du_bc = mx.squeeze(du_bc, axis=-1)

            if callable(self.target):
                target_vals = self.target(ctx.x_boundary)
            else:
                target_vals = mx.full((ctx.x_boundary.shape[0],), self.target)

            return mx.mean((du_bc - target_vals) ** 2)

        elif self.bc_type == "periodic":
            # u(left) = u(right)
            # Assumes boundary points are ordered: first half left, second half right
            n = ctx.x_boundary.shape[0] // 2
            if n > 0:
                u_left = u_bc[:n]
                u_right = u_bc[n:2*n]
                return mx.mean((u_left - u_right) ** 2)

            return mx.array(0.0)

        else:
            return mx.array(0.0)


class NormalizationLoss(LossTerm):
    """Normalization loss: (∫|u|² - 1)².

    Enforces wavefunction normalization for quantum problems.

    Args:
        weight: Loss weight.
        target: Target norm value (default: 1.0).
    """

    def __init__(self, weight: float = 10.0, target: float = 1.0, **kwargs):
        super().__init__(weight, **kwargs)
        self.target = target

    def compute(self, ctx: LossContext) -> mx.array:
        if ctx.x_interior is None:
            return mx.array(0.0)

        u = ctx.get_u(ctx.x_interior)
        if len(u.shape) > 1:
            u = mx.squeeze(u, axis=-1)

        # Approximate integral using Monte Carlo
        volume = _get_domain_volume(ctx.domain)

        integral = mx.mean(u ** 2) * volume
        return (integral - self.target) ** 2


class NonTrivialLoss(LossTerm):
    """Non-trivial solution loss: exp(-scale * ∫|u|²).

    Prevents the trivial u=0 solution by penalizing small norms.
    As u → 0, this loss → 1 (large), pushing the solution away from zero.
    As ∫|u|² grows, this loss → 0 exponentially.

    Args:
        weight: Loss weight.
        scale: Exponential decay scale (default: 10.0).
    """

    def __init__(self, weight: float = 1.0, scale: float = 10.0, **kwargs):
        super().__init__(weight, **kwargs)
        self.scale = scale

    def compute(self, ctx: LossContext) -> mx.array:
        if ctx.x_interior is None:
            return mx.array(0.0)

        u = ctx.get_u(ctx.x_interior)
        if len(u.shape) > 1:
            u = mx.squeeze(u, axis=-1)

        volume = _get_domain_volume(ctx.domain)
        u2_integral = mx.mean(u ** 2) * volume
        return mx.exp(-self.scale * u2_integral)


class EigenvalueLoss(LossTerm):
    """Eigenvalue loss for eigenvalue problems.

    Supports three methods:
    1. Rayleigh quotient: Compute E from solution using E = <u|H|u>/<u|u>
    2. Trainable parameter: E as a separate trainable variable (positivity only)
    3. Rayleigh guide: E is trainable, guided toward Rayleigh quotient estimate
       via (E - stop_gradient(E_rayleigh))². This is the recommended method
       for eigenvalue problems with trainable E.

    The computed eigenvalue is stored in ctx.computed_eigenvalue for
    use by PDEResidualLoss.

    Args:
        weight: Loss weight.
        method: "rayleigh", "trainable", or "rayleigh_guide".
        rayleigh_fn: Custom Rayleigh quotient function (for "rayleigh"/"rayleigh_guide").
        param_name: Parameter name for trainable eigenvalue (default: "E").
    """

    def __init__(
        self,
        weight: float = 1.0,
        method: str = "rayleigh",
        rayleigh_fn: Optional[Callable] = None,
        param_name: str = "E",
        **kwargs
    ):
        super().__init__(weight, **kwargs)
        self.method = method.lower()
        self.rayleigh_fn = rayleigh_fn
        self.param_name = param_name

    def compute(self, ctx: LossContext) -> mx.array:
        if self.method == "rayleigh":
            # Compute eigenvalue via Rayleigh quotient
            if self.rayleigh_fn is not None:
                E = self.rayleigh_fn(ctx.trainer or ctx.model, ctx.x_interior, ctx.params)
            else:
                E = self._default_rayleigh(ctx)

            # Store for use by PDEResidualLoss
            ctx.computed_eigenvalue = E
            return E

        elif self.method == "trainable":
            # Get trainable eigenvalue and compute positivity loss
            if self.param_name in ctx.trainable_params:
                E = ctx.trainable_params[self.param_name]
                # Store eigenvalue (extract scalar if needed)
                ctx.computed_eigenvalue = E[0] if E.shape else E
                # Return positivity loss: e^(-10|E|) encourages E > 0
                # This loss goes to 0 as |E| increases, allowing other losses to drive E
                return mx.exp(-10.0 * mx.abs(E[0] if E.shape else E))
            elif self.param_name in ctx.params:
                # Fallback to fixed param
                E = mx.array(ctx.params[self.param_name])
                ctx.computed_eigenvalue = E
                return mx.exp(-10.0 * mx.abs(E))
            return mx.array(0.0)

        elif self.method == "rayleigh_guide":
            # Compute Rayleigh quotient and guide trainable E toward it.
            # Like PyKAN: energy_loss = (E - E_rayleigh.detach()) ** 2
            if self.param_name not in ctx.trainable_params:
                return mx.array(0.0)

            E = ctx.trainable_params[self.param_name]
            E_val = E[0] if E.shape else E
            ctx.computed_eigenvalue = E_val

            # Compute Rayleigh quotient estimate
            if self.rayleigh_fn is not None:
                E_ray = self.rayleigh_fn(
                    ctx.trainer or ctx.model, ctx.x_interior, ctx.params
                )
            else:
                E_ray = self._default_rayleigh(ctx)

            # Guide E toward Rayleigh estimate (detach Rayleigh from E grad)
            return (E_val - mx.stop_gradient(E_ray)) ** 2

        return mx.array(0.0)

    def _default_rayleigh(self, ctx: LossContext) -> mx.array:
        """Auto-dispatch Rayleigh quotient based on detected PDE type.

        Inspects the equation string to dispatch to the correct estimator:
        - Wigner radial PDE (contains 'Derivative(W, r2'): E = ∫r²W / (2∫W)
        - Schrödinger (default): E = (∫|∇ψ|²/2 + ∫V|ψ|²) / ∫|ψ|²

        Override with rayleigh_fn for custom PDEs.
        """
        if ctx.compiled_residual is not None:
            eq = getattr(ctx.compiled_residual.parsed, "original", "") or ""
            if "Derivative(W, r2" in eq or "4*r2*Derivative" in eq:
                return self._wigner_rayleigh(ctx)
        return self._schrodinger_rayleigh(ctx)

    def _wigner_rayleigh(self, ctx: LossContext) -> mx.array:
        """Rayleigh quotient for Wigner radial PDE.

        Derived from (r² - 2E)W = ... by integrating over the domain:
        E = ∫r²W dr² / (2 ∫W dr²)
        """
        if ctx.x_interior is None:
            return mx.array(0.0)

        u = ctx.get_u(ctx.x_interior)
        if len(u.shape) > 1:
            u = mx.squeeze(u, axis=-1)

        r2 = mx.squeeze(ctx.x_interior, axis=-1)
        return mx.mean(r2 * u) / (2.0 * mx.mean(u) + 1e-8)

    def _schrodinger_rayleigh(self, ctx: LossContext) -> mx.array:
        """Rayleigh quotient for Schrödinger-type equations.

        E = <ψ|H|ψ> / <ψ|ψ> ≈ ∫(|∇ψ|²/2 + V|ψ|²) / ∫|ψ|²
        """
        if ctx.x_interior is None:
            return mx.array(0.0)

        u = ctx.get_u(ctx.x_interior)
        du = ctx.get_du(ctx.x_interior)

        # Squeeze only a trailing singleton axis. du is (N, dim) for multi-D and
        # MUST keep its coordinate axis for the kinetic sum below — squeezing a
        # size-2 axis would crash.
        if len(u.shape) > 1 and u.shape[-1] == 1:
            u = mx.squeeze(u, axis=-1)
        if len(du.shape) > 1 and du.shape[-1] == 1:
            du = mx.squeeze(du, axis=-1)

        # Kinetic energy: ½⟨|∇ψ|²⟩ = ½ ⟨ Σ_i (∂ψ/∂x_i)² ⟩.
        # For multi-D, du is (N, dim): sum the squared components over the
        # coordinate axis FIRST, then average over the batch. Using mean(du**2)
        # directly would average over N*dim and undercount the kinetic energy by
        # a factor of `dim`, biasing the eigenvalue low.
        if len(du.shape) > 1 and du.shape[-1] > 1:
            grad_sq = mx.sum(du ** 2, axis=-1)  # (N,)
        else:
            grad_sq = du ** 2
        kinetic = 0.5 * mx.mean(grad_sq)

        # Potential energy: V|ψ|² (assume harmonic oscillator V = x²/2)
        x = ctx.x_interior
        if x.shape[1] == 1:
            V = 0.5 * mx.squeeze(x, axis=-1) ** 2
        else:
            V = 0.5 * mx.sum(x ** 2, axis=-1)
        potential = mx.mean(V * u ** 2)

        # Norm
        norm = mx.mean(u ** 2) + 1e-8

        return (kinetic + potential) / norm


class SmoothnessLoss(LossTerm):
    """Smoothness loss: penalize large second derivatives.

    Encourages smooth solutions by penalizing oscillations.

    Args:
        weight: Loss weight.
        order: Derivative order to penalize (default: 2).
    """

    def __init__(self, weight: float = 0.1, order: int = 2, **kwargs):
        super().__init__(weight, **kwargs)
        self.order = order

    def compute(self, ctx: LossContext) -> mx.array:
        if ctx.x_interior is None:
            return mx.array(0.0)

        if self.order == 2:
            d2u = ctx.get_d2u(ctx.x_interior)
        else:
            d2u = ctx.get_du(ctx.x_interior)

        if len(d2u.shape) > 1:
            d2u = mx.squeeze(d2u, axis=-1)

        return mx.mean(d2u ** 2)


class DecayLoss(LossTerm):
    """Decay loss: penalize large values near boundaries.

    Useful for problems where the solution should decay to zero
    at the domain boundaries.

    Args:
        weight: Loss weight.
        threshold_ratio: Fraction of domain where decay is enforced.
    """

    def __init__(self, weight: float = 1.0, threshold_ratio: float = 0.8, **kwargs):
        super().__init__(weight, **kwargs)
        self.threshold_ratio = threshold_ratio

    def compute(self, ctx: LossContext) -> mx.array:
        if ctx.x_interior is None or ctx.domain is None:
            return mx.array(0.0)

        u = ctx.get_u(ctx.x_interior)
        if len(u.shape) > 1:
            u = mx.squeeze(u, axis=-1)

        # Identify points near boundary
        x = ctx.x_interior

        # For each dimension, check if point is near boundary
        near_boundary = mx.zeros(x.shape[0], dtype=mx.bool_)

        for i, (lo, hi) in enumerate(ctx.domain.bounds):
            range_size = hi - lo
            threshold_dist = range_size * (1 - self.threshold_ratio) / 2

            near_lo = x[:, i] < (lo + threshold_dist)
            near_hi = x[:, i] > (hi - threshold_dist)
            near_boundary = near_boundary | near_lo | near_hi

        # Penalize large values near boundary
        u_boundary = mx.where(near_boundary, u ** 2, mx.zeros_like(u))
        return mx.mean(u_boundary)


class AnchorLoss(LossTerm):
    """Anchor loss: u(x₀) = target.

    Fixes the solution value at a specific point. Useful for
    removing arbitrary constants or setting normalization.

    Args:
        weight: Loss weight.
        x0: Anchor point(s), shape (1, dim) or (k, dim).
        target: Target value(s) at anchor point(s).
    """

    def __init__(
        self,
        weight: float = 10.0,
        x0: Union[mx.array, List, np.ndarray] = None,
        target: Union[float, mx.array] = 0.0,
        **kwargs
    ):
        super().__init__(weight, **kwargs)
        if x0 is None:
            x0 = mx.array([[0.0]])
        elif not isinstance(x0, mx.array):
            x0 = mx.array(x0)
        if len(x0.shape) == 1:
            x0 = x0.reshape(1, -1)
        self.x0 = x0

        if isinstance(target, (int, float)):
            self.target = mx.array([target])
        else:
            self.target = mx.array(target)

    def compute(self, ctx: LossContext) -> mx.array:
        u0 = ctx.get_u(self.x0)
        if len(u0.shape) > 1:
            u0 = mx.squeeze(u0, axis=-1)

        return mx.mean((u0 - self.target) ** 2)


class RegularizationLoss(LossTerm):
    """Regularization loss on model parameters.

    Penalizes large coefficient values to prevent overfitting.

    Args:
        weight: Loss weight.
        norm: Norm type ("l1" or "l2").
    """

    def __init__(self, weight: float = 0.01, norm: str = "l1", **kwargs):
        super().__init__(weight, **kwargs)
        self.norm = norm.lower()

    def compute(self, ctx: LossContext) -> mx.array:
        if ctx.model is None:
            return mx.array(0.0)

        reg = mx.array(0.0)

        # Iterate over model layers
        if hasattr(ctx.model, 'layers'):
            for layer in ctx.model.layers:
                coefs = []
                if hasattr(layer, 'coef') and layer.coef.size > 0:
                    coefs.append(layer.coef)
                # Include slot coefs for mixed-basis layers
                s = 0
                while hasattr(layer, f'slot_coef_{s}'):
                    sc = getattr(layer, f'slot_coef_{s}')
                    if sc is not None and sc.size > 0:
                        coefs.append(sc)
                    s += 1
                for c in coefs:
                    if self.norm == "l1":
                        reg = reg + mx.mean(mx.abs(c))
                    else:
                        reg = reg + mx.mean(c ** 2)

        return reg


class DataLoss(LossTerm):
    """Data fitting loss: ||u(x_data) - y_data||².

    Fits the model to observed data points.

    Args:
        weight: Loss weight.
        x_data: Data input points.
        y_data: Data output values.
    """

    def __init__(
        self,
        weight: float = 1.0,
        x_data: Optional[mx.array] = None,
        y_data: Optional[mx.array] = None,
        **kwargs
    ):
        super().__init__(weight, **kwargs)
        self.x_data = x_data
        self.y_data = y_data

    def compute(self, ctx: LossContext) -> mx.array:
        if self.x_data is None or self.y_data is None:
            return mx.array(0.0)

        u_pred = ctx.get_u(self.x_data)
        if len(u_pred.shape) > 1:
            u_pred = mx.squeeze(u_pred, axis=-1)

        y_target = self.y_data
        if len(y_target.shape) > 1:
            y_target = mx.squeeze(y_target, axis=-1)

        return mx.mean((u_pred - y_target) ** 2)


# =============================================================================
# LOSS COMPOSER
# =============================================================================

class LossComposer:
    """Compose multiple loss terms with weights.

    The composer manages a collection of loss terms and computes
    the total weighted loss. Individual terms can be enabled/disabled
    or have their weights adjusted.

    Example:
        >>> composer = LossComposer()
        >>> composer.add(PDEResidualLoss(weight=1.0))
        >>> composer.add(BoundaryConditionLoss(weight=10.0))
        >>> composer.add(NormalizationLoss(weight=5.0))

        >>> total, breakdown = composer.compute_total(ctx)
        >>> print(f"Total: {total}, Breakdown: {breakdown}")
    """

    def __init__(self):
        self.terms: List[LossTerm] = []

    def add(self, term: LossTerm) -> "LossComposer":
        """Add a loss term.

        Args:
            term: Loss term to add.

        Returns:
            Self for chaining.
        """
        self.terms.append(term)
        return self

    def remove(self, term_id: str) -> "LossComposer":
        """Remove a loss term by ID.

        Args:
            term_id: ID of term to remove.

        Returns:
            Self for chaining.
        """
        self.terms = [t for t in self.terms if t.id != term_id]
        return self

    def get(self, term_id: str) -> Optional[LossTerm]:
        """Get a loss term by ID."""
        for t in self.terms:
            if t.id == term_id:
                return t
        return None

    def set_weight(self, term_id: str, weight: float) -> "LossComposer":
        """Set weight for a loss term.

        Args:
            term_id: ID of term.
            weight: New weight value.

        Returns:
            Self for chaining.
        """
        for t in self.terms:
            if t.id == term_id:
                t.weight = weight
                break
        return self

    def set_weights(self, weights: Dict[str, float]) -> "LossComposer":
        """Set multiple weights at once.

        Args:
            weights: Dict mapping term IDs to weights.

        Returns:
            Self for chaining.
        """
        for term_id, weight in weights.items():
            self.set_weight(term_id, weight)
        return self

    def enable(self, term_id: str) -> "LossComposer":
        """Enable a loss term.

        Args:
            term_id: ID of term to enable.

        Returns:
            Self for chaining.
        """
        for t in self.terms:
            if t.id == term_id:
                t.enabled = True
                break
        return self

    def disable(self, term_id: str) -> "LossComposer":
        """Disable a loss term.

        Args:
            term_id: ID of term to disable.

        Returns:
            Self for chaining.
        """
        for t in self.terms:
            if t.id == term_id:
                t.enabled = False
                break
        return self

    def enable_only(self, term_ids: Set[str]) -> "LossComposer":
        """Enable only specified terms, disable all others.

        Args:
            term_ids: Set of term IDs to enable.

        Returns:
            Self for chaining.
        """
        for t in self.terms:
            t.enabled = t.id in term_ids
        return self

    def compute_total(
        self, ctx: LossContext, breakdown: bool = True
    ) -> Tuple[mx.array, Dict[str, float]]:
        """Compute total weighted loss.

        Args:
            ctx: Loss context with all required data.
            breakdown: If True (default), also return a per-term ``{id: float}``
                breakdown for logging. Building it calls ``float()`` on each
                term, which forces a host-device sync — so pass ``breakdown=False``
                on the differentiated training step (the gradient does not need
                the breakdown) to keep the whole loss on the GPU. When False the
                second return value is an empty dict.

        Returns:
            Tuple of (total_loss, breakdown_dict).
        """
        total = mx.array(0.0)
        bd: Dict[str, float] = {}

        for term in self.terms:
            if term.enabled:
                value = term.compute(ctx)
                total = total + term.weight * value
                if breakdown:
                    bd[term.id] = float(value)

        return total, bd

    def build_loss_fn(self) -> Callable[[LossContext], mx.array]:
        """Build a loss function from this composer.

        Returns:
            Function (ctx) -> total_loss.
        """
        def loss_fn(ctx: LossContext) -> mx.array:
            total, _ = self.compute_total(ctx)
            return total

        return loss_fn

    def summary(self) -> str:
        """Get a summary of all loss terms."""
        lines = ["Loss Terms:"]
        for t in self.terms:
            status = "✓" if t.enabled else "✗"
            lines.append(f"  {status} {t.id}: weight={t.weight}")
        return '\n'.join(lines)

    def __repr__(self):
        enabled = sum(1 for t in self.terms if t.enabled)
        return f"LossComposer({len(self.terms)} terms, {enabled} enabled)"


# =============================================================================
# DEFAULT LOSS CONFIGURATIONS
# =============================================================================

def default_pinn_losses() -> LossComposer:
    """Create default loss configuration for PINN problems."""
    composer = LossComposer()
    composer.add(PDEResidualLoss(weight=1.0))
    composer.add(BoundaryConditionLoss(weight=10.0))
    return composer


def default_eigenvalue_losses() -> LossComposer:
    """Create default loss configuration for eigenvalue problems."""
    composer = LossComposer()
    composer.add(PDEResidualLoss(weight=10.0))
    composer.add(NormalizationLoss(weight=10.0))
    composer.add(NonTrivialLoss(weight=20.0))
    composer.add(EigenvalueLoss(weight=1.0, method="rayleigh"))
    return composer


def default_wigner_losses() -> LossComposer:
    """Create default loss configuration for Wigner function problems."""
    composer = LossComposer()
    composer.add(PDEResidualLoss(weight=10.0, normalize=True))
    composer.add(NormalizationLoss(weight=10.0))
    composer.add(NonTrivialLoss(weight=20.0))
    composer.add(DecayLoss(weight=10.0, threshold_ratio=0.8))
    composer.add(SmoothnessLoss(weight=0.5))
    composer.add(EigenvalueLoss(weight=1.0, method="rayleigh"))
    return composer


def default_quantum_losses() -> LossComposer:
    """Create default loss configuration for quantum mechanics problems."""
    composer = LossComposer()
    composer.add(PDEResidualLoss(weight=1.0))
    composer.add(BoundaryConditionLoss(weight=10.0))
    composer.add(NormalizationLoss(weight=5.0))
    composer.add(NonTrivialLoss(weight=10.0))
    return composer
