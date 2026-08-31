"""PDE Expression Intermediate Representation.

Defines the AST (Abstract Syntax Tree) for PDE expressions with SymPy backing.
This IR enables parsing, analysis, and compilation of arbitrary PDEs.

Example:
    # The Schrödinger equation: -∇²ψ/2 + Vψ = Eψ
    # Becomes:
    BinaryOp(
        op="-",
        left=BinaryOp(
            op="+",
            left=BinaryOp(op="*", left=Constant(-0.5), right=Laplacian(Variable("psi"))),
            right=BinaryOp(op="*", left=FunctionCall("V", [Coordinate("x")]), right=Variable("psi"))
        ),
        right=BinaryOp(op="*", left=Parameter("E"), right=Variable("psi"))
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from abc import ABC

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None


# =============================================================================
# BASE CLASS
# =============================================================================

@dataclass
class PDEExpr(ABC):
    """Base class for PDE expression nodes.

    All expression nodes can optionally store a SymPy expression for
    symbolic manipulation, differentiation, and simplification.
    """
    # Use init=False so this doesn't interfere with subclass positional args
    _sympy_expr: Optional[Any] = field(default=None, repr=False, compare=False, init=False)

    def to_sympy(self, symbols: Optional[Dict[str, Any]] = None) -> Any:
        """Convert this expression to a SymPy expression.

        Args:
            symbols: Optional dict mapping names to SymPy symbols/functions.
                    If not provided, symbols are created automatically.

        Returns:
            SymPy expression representing this PDEExpr.
        """
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for symbolic operations. Install with: pip install sympy")

        if self._sympy_expr is not None:
            return self._sympy_expr

        symbols = symbols or {}
        return self._to_sympy_impl(symbols)

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        """Implementation of SymPy conversion. Override in subclasses."""
        raise NotImplementedError(f"{self.__class__.__name__} must implement _to_sympy_impl")

    @classmethod
    def from_sympy(cls, expr: Any, coord_names: Optional[List[str]] = None) -> "PDEExpr":
        """Create a PDEExpr from a SymPy expression.

        Args:
            expr: SymPy expression to convert.
            coord_names: List of coordinate variable names (e.g., ['x', 't']).

        Returns:
            PDEExpr tree representing the SymPy expression.
        """
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for symbolic operations.")

        coord_names = coord_names or ['x', 't', 'a', 'r', 'r2']
        return _sympy_to_pde_expr(expr, coord_names)


# =============================================================================
# ATOMIC EXPRESSIONS
# =============================================================================

@dataclass
class Variable(PDEExpr):
    """Unknown function to solve for: u, ψ, φ, W, etc.

    Variables are the unknowns in the PDE that we're trying to find.
    They are functions of the coordinates.

    Example:
        Variable("psi")  # ψ(x) in Schrödinger equation
        Variable("W")    # W(r²) in Wigner equation
    """
    name: str = ""

    def __post_init__(self):
        # Normalize common physics notation
        if self.name:
            self.name = _normalize_variable_name(self.name)

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        if self.name in symbols:
            return symbols[self.name]
        # Create as a SymPy Function (since variables are functions of coordinates)
        return sp.Function(self.name)


@dataclass
class Coordinate(PDEExpr):
    """Independent variable: x, t, a, r, r², etc.

    Coordinates are the independent variables that the unknown
    function depends on.

    Example:
        Coordinate("x")   # Spatial coordinate
        Coordinate("t")   # Time coordinate
        Coordinate("r2")  # r² = x² + p² (radial in phase space)
    """
    name: str = ""

    def __post_init__(self):
        if self.name:
            self.name = _normalize_coordinate_name(self.name)

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        if self.name in symbols:
            return symbols[self.name]
        return sp.Symbol(self.name, real=True)


@dataclass
class Parameter(PDEExpr):
    """Physics parameter: E, ℏ, m, θ, ω, Λ, etc.

    Parameters are constants in the PDE that can be specified by the user.
    They have default values that can be overridden.

    Example:
        Parameter("E", default=0.5)      # Energy eigenvalue
        Parameter("hbar", default=1.0)   # Planck constant
        Parameter("theta", default=0.1)  # Deformation parameter
    """
    name: str = ""
    default: Optional[float] = None

    def __post_init__(self):
        if self.name:
            self.name = _normalize_parameter_name(self.name)

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        if self.name in symbols:
            return symbols[self.name]
        return sp.Symbol(self.name, real=True)


@dataclass
class Constant(PDEExpr):
    """Numeric constant: 0.5, π, 2, etc.

    Example:
        Constant(0.5)
        Constant(3.14159)
    """
    value: float = 0.0

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        return sp.Float(self.value)


# =============================================================================
# DERIVATIVE EXPRESSIONS
# =============================================================================

@dataclass
class Derivative(PDEExpr):
    """Partial derivative: ∂u/∂x, ∂²u/∂x², ∂²u/∂x∂y, etc.

    Represents derivatives of an expression with respect to coordinates.

    Args:
        expr: The expression being differentiated (usually a Variable).
        wrt: Coordinate name(s) to differentiate with respect to.
             Can be a single name or list for mixed partials.
        order: Order of differentiation (1 for first, 2 for second, etc.).
               For mixed partials, this is per-coordinate.

    Example:
        Derivative(Variable("u"), "x", 1)      # ∂u/∂x
        Derivative(Variable("u"), "x", 2)      # ∂²u/∂x²
        Derivative(Variable("psi"), "t", 1)    # ∂ψ/∂t
    """
    expr: Optional[PDEExpr] = None
    wrt: Union[str, List[str]] = ""
    order: int = 1

    def __post_init__(self):
        if isinstance(self.wrt, str):
            if self.wrt:  # Only normalize if non-empty
                self.wrt = _normalize_coordinate_name(self.wrt)
        elif isinstance(self.wrt, list):
            self.wrt = [_normalize_coordinate_name(w) for w in self.wrt]
        # If wrt is something else (like int from default), leave it alone

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        inner = self.expr.to_sympy(symbols)

        if isinstance(self.wrt, str):
            coord = symbols.get(self.wrt, sp.Symbol(self.wrt, real=True))
            # If inner is a Function, apply it to coordinates first
            if isinstance(inner, sp.FunctionClass):
                inner = inner(coord)
            return sp.Derivative(inner, coord, self.order)
        else:
            # Mixed partial
            result = inner
            for w in self.wrt:
                coord = symbols.get(w, sp.Symbol(w, real=True))
                if isinstance(result, sp.FunctionClass):
                    result = result(coord)
                result = sp.Derivative(result, coord)
            return result


@dataclass
class Laplacian(PDEExpr):
    """Laplacian operator: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ...

    The Laplacian is the sum of second derivatives over all spatial coordinates.

    Args:
        expr: Expression to apply Laplacian to.
        coords: List of spatial coordinate names. If None, inferred from context.

    Example:
        Laplacian(Variable("psi"))                    # ∇²ψ
        Laplacian(Variable("u"), coords=["x", "y"])   # ∂²u/∂x² + ∂²u/∂y²
    """
    expr: Optional[PDEExpr] = None
    coords: Optional[List[str]] = None

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        inner = self.expr.to_sympy(symbols)
        coords = self.coords or ['x']

        result = sp.Integer(0)
        for c in coords:
            coord = symbols.get(c, sp.Symbol(c, real=True))
            if isinstance(inner, sp.FunctionClass):
                inner_applied = inner(coord)
            else:
                inner_applied = inner
            result = result + sp.Derivative(inner_applied, coord, 2)

        return result


@dataclass
class Gradient(PDEExpr):
    """Gradient operator: ∇u = (∂u/∂x, ∂u/∂y, ...)

    Returns a vector of first derivatives.

    Args:
        expr: Expression to compute gradient of.
        coords: List of coordinate names.
    """
    expr: Optional[PDEExpr] = None
    coords: Optional[List[str]] = None

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        inner = self.expr.to_sympy(symbols)
        coords = self.coords or ['x']

        components = []
        for c in coords:
            coord = symbols.get(c, sp.Symbol(c, real=True))
            components.append(sp.Derivative(inner, coord))

        return sp.Matrix(components)


@dataclass
class Dalembert(PDEExpr):
    """D'Alembertian (wave operator): □u = ∂²u/∂t² - c²∇²u

    The wave operator in spacetime.

    Args:
        expr: Expression to apply d'Alembertian to.
        time_coord: Name of time coordinate (default: "t").
        spatial_coords: List of spatial coordinate names.
        c: Wave speed (default: 1).
    """
    expr: Optional[PDEExpr] = None
    time_coord: str = "t"
    spatial_coords: Optional[List[str]] = None
    c: float = 1.0

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        inner = self.expr.to_sympy(symbols)

        t = symbols.get(self.time_coord, sp.Symbol(self.time_coord, real=True))
        d2_dt2 = sp.Derivative(inner, t, 2)

        spatial = self.spatial_coords or ['x']
        laplacian = sp.Integer(0)
        for c in spatial:
            coord = symbols.get(c, sp.Symbol(c, real=True))
            laplacian = laplacian + sp.Derivative(inner, coord, 2)

        return d2_dt2 - self.c**2 * laplacian


# =============================================================================
# SPECIAL OPERATORS
# =============================================================================

@dataclass
class StarProduct(PDEExpr):
    """Moyal star product: f ⋆_θ g

    The star product for deformation quantization, implementing
    non-commutative multiplication in phase space.

    Args:
        left: Left operand.
        right: Right operand.
        theta: Deformation parameter name (default: "theta").
        order: Expansion order (2, 4, or 6).

    Example:
        StarProduct(FunctionCall("H", []), Variable("Psi"), "theta")  # H ⋆_θ Ψ
    """
    left: Optional[PDEExpr] = None
    right: Optional[PDEExpr] = None
    theta: str = "theta"
    order: int = 2

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        # Star product is complex - return symbolic representation
        f = self.left.to_sympy(symbols)
        g = self.right.to_sympy(symbols)
        theta = symbols.get(self.theta, sp.Symbol(self.theta, real=True))

        # To zeroth order: f * g
        # First order adds Poisson bracket term
        # This is a placeholder - full implementation in operators.py
        return sp.Function('star_product')(f, g, theta)


@dataclass
class Hamiltonian(PDEExpr):
    """Hamiltonian operator: Ĥ

    Represents a Hamiltonian operator, typically in quantum mechanics
    or quantum cosmology contexts.

    Args:
        name: Hamiltonian identifier (default: "H").
        args: Arguments the Hamiltonian depends on.
    """
    name: str = "H"
    args: Optional[List[PDEExpr]] = None

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        H = sp.Function(self.name)
        if self.args:
            sympy_args = [a.to_sympy(symbols) for a in self.args]
            return H(*sympy_args)
        return H


# =============================================================================
# BINARY AND UNARY OPERATIONS
# =============================================================================

@dataclass
class BinaryOp(PDEExpr):
    """Binary operation: +, -, *, /, ^

    Args:
        op: Operator string ("+", "-", "*", "/", "^" or "**").
        left: Left operand.
        right: Right operand.
    """
    op: str = ""
    left: Optional[PDEExpr] = None
    right: Optional[PDEExpr] = None

    def __post_init__(self):
        # Normalize power operator
        if self.op == "**":
            self.op = "^"

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        l = self.left.to_sympy(symbols)
        r = self.right.to_sympy(symbols)

        if self.op == "+":
            return l + r
        elif self.op == "-":
            return l - r
        elif self.op == "*":
            return l * r
        elif self.op == "/":
            return l / r
        elif self.op == "^":
            return l ** r
        else:
            raise ValueError(f"Unknown binary operator: {self.op}")


@dataclass
class UnaryOp(PDEExpr):
    """Unary operation: -, sin, cos, exp, log, sqrt, abs, etc.

    Args:
        op: Operator string.
        operand: The expression to apply the operator to.
    """
    op: str = ""
    operand: Optional[PDEExpr] = None

    # Mapping from operator names to SymPy functions
    SYMPY_MAP = {
        "-": lambda x: -x,
        "neg": lambda x: -x,
        "sin": sp.sin if SYMPY_AVAILABLE else None,
        "cos": sp.cos if SYMPY_AVAILABLE else None,
        "tan": sp.tan if SYMPY_AVAILABLE else None,
        "exp": sp.exp if SYMPY_AVAILABLE else None,
        "log": sp.log if SYMPY_AVAILABLE else None,
        "ln": sp.log if SYMPY_AVAILABLE else None,
        "sqrt": sp.sqrt if SYMPY_AVAILABLE else None,
        "abs": sp.Abs if SYMPY_AVAILABLE else None,
        "sinh": sp.sinh if SYMPY_AVAILABLE else None,
        "cosh": sp.cosh if SYMPY_AVAILABLE else None,
        "tanh": sp.tanh if SYMPY_AVAILABLE else None,
        "asin": sp.asin if SYMPY_AVAILABLE else None,
        "acos": sp.acos if SYMPY_AVAILABLE else None,
        "atan": sp.atan if SYMPY_AVAILABLE else None,
    }

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        inner = self.operand.to_sympy(symbols)

        if self.op in self.SYMPY_MAP:
            fn = self.SYMPY_MAP[self.op]
            if fn is None:
                raise ImportError("SymPy required for this operation")
            return fn(inner)
        else:
            # Unknown function - represent symbolically
            return sp.Function(self.op)(inner)


@dataclass
class FunctionCall(PDEExpr):
    """Function call: V(x), H(a), f(x, y), etc.

    Represents user-defined or potential functions.

    Args:
        name: Function name.
        args: List of argument expressions.
    """
    name: str = ""
    args: List[PDEExpr] = field(default_factory=list)

    def _to_sympy_impl(self, symbols: Dict[str, Any]) -> Any:
        fn = sp.Function(self.name)
        sympy_args = [a.to_sympy(symbols) for a in self.args]
        return fn(*sympy_args) if sympy_args else fn


# =============================================================================
# PARSED EQUATION CONTAINER
# =============================================================================

@dataclass
class ParsedPDE:
    """Container for a fully parsed PDE.

    Attributes:
        lhs: Left-hand side expression.
        rhs: Right-hand side expression.
        lhs_sympy: SymPy expression for LHS (optional).
        rhs_sympy: SymPy expression for RHS (optional).
        original: Original equation string.
        format: Detected or specified format.
        residual_expr: LHS - RHS (for residual computation).
    """
    lhs: Optional[PDEExpr] = None
    rhs: Optional[PDEExpr] = None
    lhs_sympy: Optional[Any] = None
    rhs_sympy: Optional[Any] = None
    original: str = ""
    format: str = "unknown"

    @property
    def residual_expr(self) -> PDEExpr:
        """Get residual expression: LHS - RHS = 0."""
        return BinaryOp("-", self.lhs, self.rhs)

    @property
    def residual_sympy(self) -> Any:
        """Get residual as SymPy expression."""
        if self.lhs_sympy is not None and self.rhs_sympy is not None:
            return self.lhs_sympy - self.rhs_sympy
        return self.residual_expr.to_sympy()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _normalize_variable_name(name: str) -> str:
    """Normalize variable names to canonical form."""
    # Greek letter mappings
    GREEK = {
        'ψ': 'psi', 'psi': 'psi', '\\psi': 'psi',
        'Ψ': 'Psi', 'Psi': 'Psi', '\\Psi': 'Psi',
        'φ': 'phi', 'phi': 'phi', '\\phi': 'phi',
        'Φ': 'Phi', 'Phi': 'Phi', '\\Phi': 'Phi',
        'χ': 'chi', 'chi': 'chi', '\\chi': 'chi',
    }
    return GREEK.get(name, name)


def _normalize_coordinate_name(name: str) -> str:
    """Normalize coordinate names."""
    # Handle special cases
    COORDS = {
        'r²': 'r2', 'r^2': 'r2', 'r**2': 'r2',
        'τ': 'tau', 'η': 'eta',
    }
    return COORDS.get(name, name)


def _normalize_parameter_name(name: str) -> str:
    """Normalize parameter names."""
    PARAMS = {
        'ℏ': 'hbar', '\\hbar': 'hbar', 'h_bar': 'hbar',
        'θ': 'theta', '\\theta': 'theta',
        'ω': 'omega', '\\omega': 'omega',
        'λ': 'lambda_', '\\lambda': 'lambda_',
        'Λ': 'Lambda', '\\Lambda': 'Lambda',
        'α': 'alpha', '\\alpha': 'alpha',
        'β': 'beta', '\\beta': 'beta',
        'γ': 'gamma', '\\gamma': 'gamma',
    }
    return PARAMS.get(name, name)


def _sympy_to_pde_expr(expr: Any, coord_names: List[str]) -> PDEExpr:
    """Convert a SymPy expression to PDEExpr tree.

    Args:
        expr: SymPy expression.
        coord_names: List of coordinate variable names.

    Returns:
        Equivalent PDEExpr tree.
    """
    if not SYMPY_AVAILABLE:
        raise ImportError("SymPy is required")

    # Numbers
    if expr.is_Number:
        return Constant(float(expr))

    # Symbols
    if expr.is_Symbol:
        name = str(expr)
        if name in coord_names:
            return Coordinate(name)
        # Check if it looks like a parameter
        if name in ['E', 'hbar', 'm', 'theta', 'omega', 'Lambda', 'alpha', 'beta']:
            return Parameter(name)
        # Default to variable
        return Variable(name)

    # Derivatives
    if isinstance(expr, sp.Derivative):
        inner = _sympy_to_pde_expr(expr.expr, coord_names)
        # Extract derivative specification
        wrt_vars = []
        for var, count in expr.variable_count:
            wrt_vars.extend([str(var)] * count)

        if len(set(wrt_vars)) == 1:
            # Same variable multiple times
            return Derivative(inner, wrt_vars[0], len(wrt_vars))
        else:
            # Mixed partial
            return Derivative(inner, wrt_vars, 1)

    # Function applications
    if isinstance(expr, sp.Function):
        name = str(expr.func)
        args = [_sympy_to_pde_expr(a, coord_names) for a in expr.args]

        # Check for known functions
        if name.lower() in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs']:
            if len(args) == 1:
                return UnaryOp(name.lower(), args[0])

        # Unknown function
        if args:
            return FunctionCall(name, args)
        return Variable(name)

    # Binary operations
    if isinstance(expr, sp.Add):
        result = _sympy_to_pde_expr(expr.args[0], coord_names)
        for arg in expr.args[1:]:
            result = BinaryOp("+", result, _sympy_to_pde_expr(arg, coord_names))
        return result

    if isinstance(expr, sp.Mul):
        result = _sympy_to_pde_expr(expr.args[0], coord_names)
        for arg in expr.args[1:]:
            result = BinaryOp("*", result, _sympy_to_pde_expr(arg, coord_names))
        return result

    if isinstance(expr, sp.Pow):
        base = _sympy_to_pde_expr(expr.base, coord_names)
        exp = _sympy_to_pde_expr(expr.exp, coord_names)
        return BinaryOp("^", base, exp)

    # Negation
    if isinstance(expr, sp.Mul) and expr.args[0] == -1:
        inner = _sympy_to_pde_expr(sp.Mul(*expr.args[1:]), coord_names)
        return UnaryOp("-", inner)

    # Fallback: represent as constant if numeric, otherwise as function call
    try:
        return Constant(float(expr))
    except (TypeError, ValueError):
        return FunctionCall(str(expr), [])


# =============================================================================
# EXPRESSION UTILITIES
# =============================================================================

def walk_expr(expr: PDEExpr, visitor: callable) -> None:
    """Walk an expression tree, calling visitor on each node.

    Args:
        expr: Root expression to walk.
        visitor: Function called with each expression node.
    """
    visitor(expr)

    if isinstance(expr, (Derivative, Laplacian, Gradient, Dalembert, UnaryOp)):
        walk_expr(expr.expr if hasattr(expr, 'expr') else expr.operand, visitor)
    elif isinstance(expr, (BinaryOp, StarProduct)):
        walk_expr(expr.left, visitor)
        walk_expr(expr.right, visitor)
    elif isinstance(expr, FunctionCall):
        for arg in expr.args:
            walk_expr(arg, visitor)
    elif isinstance(expr, Hamiltonian) and expr.args:
        for arg in expr.args:
            walk_expr(arg, visitor)


def collect_nodes(expr: PDEExpr, node_type: type) -> List[PDEExpr]:
    """Collect all nodes of a specific type from an expression tree.

    Args:
        expr: Root expression.
        node_type: Type of nodes to collect.

    Returns:
        List of matching nodes.
    """
    results = []

    def visitor(node):
        if isinstance(node, node_type):
            results.append(node)

    walk_expr(expr, visitor)
    return results


def substitute(expr: PDEExpr, mapping: Dict[str, PDEExpr]) -> PDEExpr:
    """Substitute variables/parameters in an expression.

    Args:
        expr: Expression to substitute in.
        mapping: Dict mapping names to replacement expressions.

    Returns:
        New expression with substitutions applied.
    """
    if isinstance(expr, Variable) and expr.name in mapping:
        return mapping[expr.name]
    if isinstance(expr, Parameter) and expr.name in mapping:
        return mapping[expr.name]
    if isinstance(expr, Coordinate) and expr.name in mapping:
        return mapping[expr.name]

    # Recursively substitute in compound expressions
    if isinstance(expr, Derivative):
        return Derivative(substitute(expr.expr, mapping), expr.wrt, expr.order)
    if isinstance(expr, Laplacian):
        return Laplacian(substitute(expr.expr, mapping), expr.coords)
    if isinstance(expr, BinaryOp):
        return BinaryOp(expr.op, substitute(expr.left, mapping), substitute(expr.right, mapping))
    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, substitute(expr.operand, mapping))
    if isinstance(expr, FunctionCall):
        return FunctionCall(expr.name, [substitute(a, mapping) for a in expr.args])

    return expr
