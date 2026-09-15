"""PDE Residual Compiler.

Compiles PDEExpr expressions into executable residual functions
that can be used for training KAN models.

The compiler:
1. Takes a parsed PDE and jet requirements
2. Builds a residual function: R[u](x) = LHS - RHS
3. Implements efficient jet caching (compute derivatives once)
4. Supports both PINNTrainer and direct model evaluation

Example:
    >>> from kan_mlx_physics.pde.parser import parse
    >>> from kan_mlx_physics.pde.analysis import analyze
    >>> from kan_mlx_physics.pde.compiler import compile_residual

    >>> parsed = parse("-∇²ψ/2 + x²ψ/2 = Eψ")
    >>> analysis = analyze(parsed)
    >>> compiled = compile_residual(parsed, analysis.jet_req)

    >>> # Use with PINNTrainer
    >>> residual = compiled.evaluate(trainer, x, {"E": 0.5})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Union, Set
from functools import lru_cache

import mlx.core as mx

from .expr import (
    PDEExpr, ParsedPDE, Variable, Coordinate, Parameter, Constant,
    Derivative, Laplacian, Gradient, Dalembert, StarProduct, Hamiltonian,
    BinaryOp, UnaryOp, FunctionCall
)
from .analysis import JetRequirements, DerivativeSpec


# =============================================================================
# JET CACHE
# =============================================================================

@dataclass
class JetCache:
    """Cache for computed derivatives.

    Stores function values and derivatives to avoid recomputation
    when evaluating complex expressions.

    Attributes:
        values: Dict mapping variable names to their values.
        derivatives: Dict mapping (variable, coord, order) to derivative values.
        laplacians: Dict mapping variable names to Laplacian values.
        gradients: Dict mapping variable names to gradient values.
    """
    values: Dict[str, mx.array] = field(default_factory=dict)
    derivatives: Dict[tuple, mx.array] = field(default_factory=dict)
    laplacians: Dict[str, mx.array] = field(default_factory=dict)
    gradients: Dict[str, mx.array] = field(default_factory=dict)

    def get_value(self, name: str) -> Optional[mx.array]:
        """Get cached function value."""
        return self.values.get(name)

    def set_value(self, name: str, value: mx.array):
        """Cache function value."""
        self.values[name] = value

    def get_derivative(self, variable: str, wrt: str, order: int) -> Optional[mx.array]:
        """Get cached derivative."""
        key = (variable, wrt, order)
        return self.derivatives.get(key)

    def set_derivative(self, variable: str, wrt: str, order: int, value: mx.array):
        """Cache derivative value."""
        key = (variable, wrt, order)
        self.derivatives[key] = value

    def get_laplacian(self, name: str) -> Optional[mx.array]:
        """Get cached Laplacian."""
        return self.laplacians.get(name)

    def set_laplacian(self, name: str, value: mx.array):
        """Cache Laplacian value."""
        self.laplacians[name] = value

    def clear(self):
        """Clear all cached values."""
        self.values.clear()
        self.derivatives.clear()
        self.laplacians.clear()
        self.gradients.clear()


# =============================================================================
# COMPILED RESIDUAL
# =============================================================================

@dataclass
class CompiledResidual:
    """A compiled PDE residual function.

    Attributes:
        parsed: The original parsed PDE.
        jet_req: Jet requirements for this PDE.
        coord_map: Mapping from coordinate names to array indices.
        residual_expr: The residual expression (LHS - RHS).
        custom_functions: Registry of custom functions (name -> callable).
        derivative_method: Method for computing derivatives ("autodiff" or "finite_diff").
        finite_diff_h: Step size for finite differences (only used if derivative_method="finite_diff").
    """
    parsed: ParsedPDE
    jet_req: JetRequirements
    coord_map: Dict[str, int] = field(default_factory=dict)
    custom_functions: Dict[str, Callable] = field(default_factory=dict)
    derivative_method: str = "autodiff"  # Default to autodiff for accuracy
    finite_diff_h: float = 1e-4  # For backward compatibility

    def __post_init__(self):
        # Build coordinate mapping
        if not self.coord_map:
            coords = sorted(self.jet_req.coordinates)
            self.coord_map = {c: i for i, c in enumerate(coords)}

    def register_function(self, name: str, fn: Callable) -> "CompiledResidual":
        """Register a custom function for use in the equation.

        Custom functions are called during residual evaluation when the
        parser encounters a FunctionCall with the matching name.

        Args:
            name: Function name as it appears in the equation (e.g., "V", "f").
            fn: Callable taking mx.array argument(s) and returning mx.array.

        Returns:
            Self for chaining.

        Example:
            >>> compiled.register_function("V", lambda x: x**4)  # Quartic potential
            >>> compiled.register_function("f", lambda x: mx.sin(x))  # Custom forcing
        """
        self.custom_functions[name.lower()] = fn
        return self

    @property
    def residual_expr(self) -> PDEExpr:
        """Get residual expression (LHS - RHS)."""
        return self.parsed.residual_expr

    def evaluate(
        self,
        trainer_or_model: Any,
        x: mx.array,
        params: Dict[str, float],
        eigenvalue: Optional[float] = None,
    ) -> mx.array:
        """Evaluate the residual at given points.

        Args:
            trainer_or_model: A PINNTrainer or MultKAN model.
            x: Input points, shape (N, dim).
            params: Parameter values (E, hbar, etc.).
            eigenvalue: Optional eigenvalue override (for Rayleigh quotient).

        Returns:
            Residual values, shape (N,).
        """
        # Create jet cache
        cache = JetCache()

        # Compute all required jets
        self._compute_jets(trainer_or_model, x, cache, params)

        # If eigenvalue provided, override E parameter
        if eigenvalue is not None:
            params = {**params, 'E': float(eigenvalue)}

        # Evaluate residual expression
        return self._eval_expr(self.residual_expr, x, cache, params)

    def _compute_jets(
        self,
        trainer_or_model: Any,
        x: mx.array,
        cache: JetCache,
        params: Dict[str, float]
    ):
        """Compute and cache all required derivatives."""
        # Determine if we have a trainer or raw model
        has_trainer = hasattr(trainer_or_model, 'u') and hasattr(trainer_or_model, 'du')

        # Compute function values
        for var_name in self.jet_req.unknowns:
            if has_trainer:
                value = trainer_or_model.u(x)
            else:
                value = trainer_or_model(x)

            # Ensure correct shape
            if len(value.shape) > 1 and value.shape[-1] == 1:
                value = mx.squeeze(value, axis=-1)

            cache.set_value(var_name, value)

        # Compute derivatives
        for deriv_spec in self.jet_req.derivatives:
            var_name = deriv_spec.variable
            wrt = deriv_spec.wrt
            order = deriv_spec.order

            if has_trainer:
                if order == 1:
                    deriv_value = trainer_or_model.du(x)
                elif order == 2:
                    deriv_value = trainer_or_model.d2u(x)
                else:
                    # Higher order - use finite differences
                    deriv_value = self._compute_higher_derivative(
                        trainer_or_model, x, order
                    )
                # du/d2u return (N, dim) for multi-D inputs — select the column
                # for the coordinate this derivative is taken w.r.t. (wrt). For
                # 1D they already return (N,). Without this, a 2D/3D PDE would
                # multiply a (N,) term by an (N, dim) derivative and crash.
                if len(deriv_value.shape) > 1 and deriv_value.shape[-1] > 1:
                    coord_idx = self.coord_map.get(wrt, 0)
                    if coord_idx < deriv_value.shape[-1]:
                        deriv_value = deriv_value[:, coord_idx]
            else:
                # Choose derivative method based on configuration
                if self.derivative_method == "autodiff":
                    deriv_value = self._autodiff_derivative(
                        trainer_or_model, x, wrt, order
                    )
                else:
                    # Use finite differences with configured step size
                    deriv_value = self._finite_diff_derivative(
                        trainer_or_model, x, wrt, order, h=self.finite_diff_h
                    )

            # Ensure correct shape
            if len(deriv_value.shape) > 1 and deriv_value.shape[-1] == 1:
                deriv_value = mx.squeeze(deriv_value, axis=-1)

            cache.set_derivative(var_name, wrt, order, deriv_value)

        # Compute Laplacians
        for var_name in self.jet_req.needs_laplacian:
            if has_trainer and hasattr(trainer_or_model, 'laplacian'):
                lap_value = trainer_or_model.laplacian(x)
            else:
                # Sum of second derivatives
                lap_value = self._compute_laplacian(trainer_or_model, x, has_trainer)

            if len(lap_value.shape) > 1 and lap_value.shape[-1] == 1:
                lap_value = mx.squeeze(lap_value, axis=-1)

            cache.set_laplacian(var_name, lap_value)

    def _eval_expr(
        self,
        expr: PDEExpr,
        x: mx.array,
        cache: JetCache,
        params: Dict[str, float]
    ) -> mx.array:
        """Recursively evaluate an expression using cached jets."""
        if expr is None:
            return mx.zeros(x.shape[0])

        # Constant
        if isinstance(expr, Constant):
            return mx.full((x.shape[0],), expr.value)

        # Variable (unknown function)
        if isinstance(expr, Variable):
            cached = cache.get_value(expr.name)
            if cached is not None:
                return cached
            # Fallback: assume it's already been computed
            return mx.zeros(x.shape[0])

        # Coordinate
        if isinstance(expr, Coordinate):
            idx = self.coord_map.get(expr.name, 0)
            if idx < x.shape[1]:
                return x[:, idx]
            return mx.zeros(x.shape[0])

        # Parameter
        if isinstance(expr, Parameter):
            value = params.get(expr.name, expr.default or 0.0)
            # Handle mx.array values (for trainable params) - preserve gradient flow
            if isinstance(value, mx.array):
                # Broadcast the array to match batch size
                if value.shape == () or value.shape == (1,):
                    return mx.broadcast_to(value.reshape((1,)), (x.shape[0],))
                return value
            return mx.full((x.shape[0],), value)

        # Derivative
        if isinstance(expr, Derivative):
            if isinstance(expr.expr, Variable):
                var_name = expr.expr.name
                wrt = expr.wrt if isinstance(expr.wrt, str) else expr.wrt[0]
                cached = cache.get_derivative(var_name, wrt, expr.order)
                if cached is not None:
                    return cached

            # Fallback: evaluate inner expression
            return mx.zeros(x.shape[0])

        # Laplacian
        if isinstance(expr, Laplacian):
            if isinstance(expr.expr, Variable):
                cached = cache.get_laplacian(expr.expr.name)
                if cached is not None:
                    return cached
            return mx.zeros(x.shape[0])

        # Binary operations
        if isinstance(expr, BinaryOp):
            left = self._eval_expr(expr.left, x, cache, params)
            right = self._eval_expr(expr.right, x, cache, params)

            if expr.op == '+':
                return left + right
            elif expr.op == '-':
                return left - right
            elif expr.op == '*':
                return left * right
            elif expr.op == '/':
                return left / (right + 1e-10)  # Avoid division by zero
            elif expr.op == '^':
                return mx.power(left, right)
            else:
                raise ValueError(f"Unknown binary operator: {expr.op}")

        # Unary operations
        if isinstance(expr, UnaryOp):
            operand = self._eval_expr(expr.operand, x, cache, params)

            op = expr.op.lower()
            if op == '-' or op == 'neg':
                return -operand
            elif op == 'sin':
                return mx.sin(operand)
            elif op == 'cos':
                return mx.cos(operand)
            elif op == 'tan':
                return mx.tan(operand)
            elif op == 'exp':
                return mx.exp(operand)
            elif op == 'log' or op == 'ln':
                return mx.log(mx.maximum(operand, 1e-10))
            elif op == 'sqrt':
                return mx.sqrt(mx.maximum(operand, 0.0))
            elif op == 'abs':
                return mx.abs(operand)
            elif op == 'sinh':
                return mx.sinh(operand)
            elif op == 'cosh':
                return mx.cosh(operand)
            elif op == 'tanh':
                return mx.tanh(operand)
            else:
                raise ValueError(f"Unknown unary operator: {expr.op}")

        # Function call
        if isinstance(expr, FunctionCall):
            # Evaluate arguments
            args = [self._eval_expr(arg, x, cache, params) for arg in expr.args]

            # Check for custom registered functions first
            name = expr.name.lower()
            if name in self.custom_functions:
                fn = self.custom_functions[name]
                try:
                    if len(args) == 1:
                        return fn(args[0])
                    else:
                        return fn(*args)
                except Exception as e:
                    # Fallback if custom function fails
                    import warnings
                    warnings.warn(f"Custom function '{name}' failed: {e}")
                    return mx.zeros(x.shape[0])

            # Unknown function: FAIL LOUDLY. Previously an unregistered V(x) was
            # silently replaced by the harmonic-oscillator potential 0.5*x**2 (and
            # U(a) by a**3), so a user solving a different potential trained against
            # the wrong physics with only a warning. Registering the function is
            # mandatory — an unknown callable in the equation is an error, not a
            # default. (Presets that *want* a harmonic default should register it
            # explicitly via .function('V', lambda x: 0.5*x**2).)
            raise ValueError(
                f"Unknown function '{expr.name}' in the equation. Register it with "
                f".function('{expr.name}', fn) before solving. Refusing to substitute "
                f"a default potential, which would silently solve a different problem."
            )

        # Star product (placeholder - full implementation in operators)
        if isinstance(expr, StarProduct):
            left = self._eval_expr(expr.left, x, cache, params)
            right = self._eval_expr(expr.right, x, cache, params)
            # To zeroth order, star product is just multiplication
            return left * right

        # Gradient, d'Alembertian - handled via cache
        if isinstance(expr, (Gradient, Dalembert)):
            # These should be pre-computed in cache
            return mx.zeros(x.shape[0])

        # Hamiltonian
        if isinstance(expr, Hamiltonian):
            # Hamiltonian is typically defined by the full expression
            return mx.zeros(x.shape[0])

        # Fallback
        return mx.zeros(x.shape[0])

    def _compute_higher_derivative(
        self,
        trainer: Any,
        x: mx.array,
        order: int,
        h: float = 1e-4
    ) -> mx.array:
        """Compute higher-order derivatives using finite differences."""
        if order == 3:
            # Third derivative: (f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)) / (2h³)
            xp2 = x + 2*h
            xp1 = x + h
            xm1 = x - h
            xm2 = x - 2*h

            fp2 = trainer.u(xp2)
            fp1 = trainer.u(xp1)
            fm1 = trainer.u(xm1)
            fm2 = trainer.u(xm2)

            return (fp2 - 2*fp1 + 2*fm1 - fm2) / (2 * h**3)

        elif order == 4:
            # Fourth derivative: 5-point stencil
            xp2 = x + 2*h
            xp1 = x + h
            x0 = x
            xm1 = x - h
            xm2 = x - 2*h

            fp2 = trainer.u(xp2)
            fp1 = trainer.u(xp1)
            f0 = trainer.u(x0)
            fm1 = trainer.u(xm1)
            fm2 = trainer.u(xm2)

            return (fp2 - 4*fp1 + 6*f0 - 4*fm1 + fm2) / (h**4)

        else:
            # General case: use second derivative recursively
            return trainer.d2u(x)

    def _finite_diff_derivative(
        self,
        model: Any,
        x: mx.array,
        wrt: str,
        order: int,
        h: float = 1e-4
    ) -> mx.array:
        """Compute derivative using finite differences on raw model."""
        idx = self.coord_map.get(wrt, 0)

        if order == 1:
            # Central difference
            x_plus = x.at[:, idx].add(h)
            x_minus = x.at[:, idx].add(-h)

            f_plus = model(x_plus)
            f_minus = model(x_minus)

            return (f_plus - f_minus) / (2 * h)

        elif order == 2:
            # Second derivative
            x_plus = x.at[:, idx].add(h)
            x_minus = x.at[:, idx].add(-h)

            f_plus = model(x_plus)
            f_0 = model(x)
            f_minus = model(x_minus)

            return (f_plus - 2*f_0 + f_minus) / (h**2)

        else:
            raise NotImplementedError(f"Order {order} derivatives not implemented")

    def _autodiff_derivative(
        self,
        model: Any,
        x: mx.array,
        wrt: str,
        order: int
    ) -> mx.array:
        """Compute derivative using automatic differentiation.

        Uses the batch-grad sum trick: differentiate sum(f(x)) to get per-sample gradients.
        This is accurate and avoids float32 cancellation issues in finite differences.

        Args:
            model: The model to differentiate
            x: Input points (N, dim)
            wrt: Which coordinate to differentiate with respect to
            order: Derivative order (1 or 2)

        Returns:
            Derivatives at each point (N,)
        """
        idx = self.coord_map.get(wrt, 0)

        # Model output (squeeze to (N,) if needed)
        def u_vec(x_batch: mx.array) -> mx.array:
            y = model(x_batch)
            if len(y.shape) > 1 and y.shape[-1] == 1:
                y = mx.squeeze(y, axis=-1)
            return y

        # Sum over batch (required by mx.grad)
        def u_sum(x_batch: mx.array) -> mx.array:
            return mx.sum(u_vec(x_batch))

        if order == 1:
            # First derivative: d(sum u)/dx
            grad_fn = mx.grad(u_sum)
            grad_full = grad_fn(x)  # (N, dim)

            # Extract derivative w.r.t. specific coordinate
            if len(grad_full.shape) == 1:
                return grad_full  # 1D case
            else:
                return grad_full[:, idx]  # Multi-D case

        elif order == 2:
            # Second derivative: d²(sum u)/dx²
            # First compute d(sum u)/dx
            grad_fn = mx.grad(u_sum)

            # Then differentiate the gradient sum w.r.t. x
            def grad_sum(x_batch: mx.array) -> mx.array:
                g = grad_fn(x_batch)
                if len(g.shape) == 1:
                    return mx.sum(g)
                else:
                    return mx.sum(g[:, idx])

            grad2_fn = mx.grad(grad_sum)
            grad2_full = grad2_fn(x)  # (N, dim)

            # Extract second derivative w.r.t. specific coordinate
            if len(grad2_full.shape) == 1:
                return grad2_full
            else:
                return grad2_full[:, idx]

        else:
            raise NotImplementedError(f"Order {order} autodiff derivatives not implemented")

    def _compute_laplacian(
        self,
        trainer_or_model: Any,
        x: mx.array,
        has_trainer: bool
    ) -> mx.array:
        """Compute Laplacian as sum of second derivatives."""
        dim = x.shape[1]
        result = mx.zeros(x.shape[0])

        for i in range(dim):
            if has_trainer:
                # Use trainer's d2u (assumes it computes ∂²u/∂x_i²)
                d2u = trainer_or_model.d2u(x)
                if len(d2u.shape) > 1:
                    d2u = mx.squeeze(d2u, axis=-1)
                result = result + d2u
            else:
                # Finite difference
                coord = list(self.coord_map.keys())[i] if i < len(self.coord_map) else f'x{i}'
                d2u = self._finite_diff_derivative(trainer_or_model, x, coord, 2)
                if len(d2u.shape) > 1:
                    d2u = mx.squeeze(d2u, axis=-1)
                result = result + d2u

        return result


# =============================================================================
# COMPILER
# =============================================================================

class ResidualCompiler:
    """Compile PDEExpr to executable residual functions.

    The compiler takes a parsed PDE and its jet requirements,
    then produces a CompiledResidual that can efficiently
    evaluate the residual for training.

    Example:
        >>> compiler = ResidualCompiler()
        >>> compiled = compiler.compile(parsed, jet_req)
        >>> residual = compiled.evaluate(trainer, x, params)
    """

    def compile(
        self,
        parsed: ParsedPDE,
        jet_req: JetRequirements,
        coord_order: Optional[List[str]] = None
    ) -> CompiledResidual:
        """Compile a parsed PDE to a residual function.

        Args:
            parsed: Parsed PDE equation.
            jet_req: Jet requirements from analysis.
            coord_order: Optional ordering of coordinates.

        Returns:
            CompiledResidual ready for evaluation.
        """
        # Build coordinate map
        if coord_order:
            coord_map = {c: i for i, c in enumerate(coord_order)}
        else:
            # Use alphabetical order, but put 't' last (common convention)
            coords = sorted(jet_req.coordinates)
            if 't' in coords:
                coords.remove('t')
                coords.append('t')
            coord_map = {c: i for i, c in enumerate(coords)}

        return CompiledResidual(
            parsed=parsed,
            jet_req=jet_req,
            coord_map=coord_map
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compile_residual(
    parsed: ParsedPDE,
    jet_req: JetRequirements,
    coord_order: Optional[List[str]] = None,
    derivative_method: str = "autodiff",
    finite_diff_h: float = 1e-4
) -> CompiledResidual:
    """Compile a parsed PDE to a residual function.

    This is a convenience function that creates a ResidualCompiler
    and compiles the PDE.

    Args:
        parsed: Parsed PDE equation.
        jet_req: Jet requirements from analysis.
        coord_order: Optional ordering of coordinates.
        derivative_method: Method for computing derivatives ("autodiff" or "finite_diff").
                          Default is "autodiff" for better accuracy.
        finite_diff_h: Step size for finite differences (only used if derivative_method="finite_diff").

    Returns:
        CompiledResidual ready for evaluation.

    Example:
        >>> from kan_mlx_physics.pde.parser import parse
        >>> from kan_mlx_physics.pde.analysis import analyze
        >>> from kan_mlx_physics.pde.compiler import compile_residual

        >>> parsed = parse("-∇²ψ/2 + x²ψ/2 = Eψ")
        >>> analysis = analyze(parsed)
        >>> compiled = compile_residual(parsed, analysis.jet_req)

        # For backward compatibility (not recommended)
        >>> compiled = compile_residual(parsed, analysis.jet_req,
        ...                             derivative_method="finite_diff", finite_diff_h=1e-4)
    """
    if derivative_method == "finite_diff":
        import warnings
        warnings.warn(
            "finite_diff is deprecated (less accurate due to float32 cancellation). "
            "Use autodiff for better accuracy.",
            DeprecationWarning,
            stacklevel=2
        )

    compiler = ResidualCompiler()
    compiled = compiler.compile(parsed, jet_req, coord_order)

    # Set derivative method configuration
    compiled.derivative_method = derivative_method
    compiled.finite_diff_h = finite_diff_h

    return compiled


def make_residual_fn(
    compiled: CompiledResidual,
    params: Dict[str, float]
) -> Callable[[Any, mx.array], mx.array]:
    """Create a residual function with fixed parameters.

    Args:
        compiled: Compiled residual.
        params: Fixed parameter values.

    Returns:
        Function (trainer, x) -> residual.

    Example:
        >>> residual_fn = make_residual_fn(compiled, {"E": 0.5, "hbar": 1.0})
        >>> residual = residual_fn(trainer, x)
    """
    def residual_fn(trainer_or_model: Any, x: mx.array) -> mx.array:
        return compiled.evaluate(trainer_or_model, x, params)

    return residual_fn


def make_loss_fn(
    compiled: CompiledResidual,
    params: Dict[str, float],
    normalize: bool = True
) -> Callable[[Any, mx.array], mx.array]:
    """Create a loss function from compiled residual.

    Args:
        compiled: Compiled residual.
        params: Fixed parameter values.
        normalize: Whether to normalize by u² norm.

    Returns:
        Function (trainer, x) -> loss.

    Example:
        >>> loss_fn = make_loss_fn(compiled, {"E": 0.5})
        >>> loss = loss_fn(trainer, x)
    """
    def loss_fn(trainer_or_model: Any, x: mx.array) -> mx.array:
        residual = compiled.evaluate(trainer_or_model, x, params)
        loss = mx.mean(residual ** 2)

        if normalize:
            # Get function value
            if hasattr(trainer_or_model, 'u'):
                u = trainer_or_model.u(x)
            else:
                u = trainer_or_model(x)

            if len(u.shape) > 1:
                u = mx.squeeze(u, axis=-1)

            u_norm = mx.mean(u ** 2) + 1e-6
            loss = loss / u_norm

        return loss

    return loss_fn
