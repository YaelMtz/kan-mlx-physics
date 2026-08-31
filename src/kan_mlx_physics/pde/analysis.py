"""PDE Equation Analysis.

Analyzes parsed PDE expressions to extract:
- Unknown functions (u, ψ, φ, W)
- Coordinates (x, t, a, r²)
- Parameters (E, ℏ, m, θ)
- Required derivatives (jet requirements)
- Equation type classification

Example:
    >>> from kan_mlx_physics.pde.parser import parse
    >>> from kan_mlx_physics.pde.analysis import analyze

    >>> parsed = parse("-∇²ψ/2 + x²ψ/2 = Eψ")
    >>> analysis = analyze(parsed)

    >>> print(analysis.unknowns)        # {'psi'}
    >>> print(analysis.coordinates)     # {'x'}
    >>> print(analysis.parameters)      # {'E': None}
    >>> print(analysis.equation_type)   # 'schrodinger'
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Callable, Any, Tuple

from .expr import (
    PDEExpr, ParsedPDE, Variable, Coordinate, Parameter, Constant,
    Derivative, Laplacian, Gradient, Dalembert, StarProduct, Hamiltonian,
    BinaryOp, UnaryOp, FunctionCall, walk_expr, collect_nodes
)


# =============================================================================
# JET REQUIREMENTS
# =============================================================================

@dataclass
class DerivativeSpec:
    """Specification for a derivative requirement.

    Attributes:
        variable: Name of the unknown being differentiated.
        wrt: Coordinate(s) to differentiate with respect to.
        order: Order of derivative.
    """
    variable: str
    wrt: str
    order: int

    def __hash__(self):
        return hash((self.variable, self.wrt, self.order))

    def __eq__(self, other):
        if not isinstance(other, DerivativeSpec):
            return False
        return (self.variable == other.variable and
                self.wrt == other.wrt and
                self.order == other.order)

    def __repr__(self):
        if self.order == 1:
            return f"∂{self.variable}/∂{self.wrt}"
        return f"∂{self.order}{self.variable}/∂{self.wrt}{self.order}"


@dataclass
class JetRequirements:
    """All derivatives needed to evaluate a PDE.

    The "jet" is the collection of a function and all its derivatives
    up to some order. This class tracks which derivatives are needed
    to compute the PDE residual.

    Attributes:
        unknowns: Set of unknown function names.
        coordinates: Set of coordinate names.
        parameters: Dict mapping parameter names to default values.
        derivatives: Set of derivative specifications.
        needs_laplacian: Set of unknowns that need Laplacian.
        needs_gradient: Set of unknowns that need gradient.
        needs_dalembert: Set of unknowns that need d'Alembertian.
        max_order: Maximum derivative order needed.
        dimension: Inferred spatial dimension.
    """
    unknowns: Set[str] = field(default_factory=set)
    coordinates: Set[str] = field(default_factory=set)
    parameters: Dict[str, Optional[float]] = field(default_factory=dict)
    derivatives: Set[DerivativeSpec] = field(default_factory=set)
    needs_laplacian: Set[str] = field(default_factory=set)
    needs_gradient: Set[str] = field(default_factory=set)
    needs_dalembert: Set[str] = field(default_factory=set)
    max_order: int = 0
    dimension: int = 1

    def add_derivative(self, variable: str, wrt: str, order: int):
        """Add a derivative requirement."""
        spec = DerivativeSpec(variable, wrt, order)
        self.derivatives.add(spec)
        self.max_order = max(self.max_order, order)

    def get_derivatives_for(self, variable: str) -> List[DerivativeSpec]:
        """Get all derivative specs for a specific variable."""
        return [d for d in self.derivatives if d.variable == variable]

    def summary(self) -> str:
        """Get a human-readable summary of requirements."""
        lines = [
            f"Unknowns: {', '.join(sorted(self.unknowns)) or 'none'}",
            f"Coordinates: {', '.join(sorted(self.coordinates)) or 'none'}",
            f"Parameters: {', '.join(f'{k}={v}' for k, v in sorted(self.parameters.items())) or 'none'}",
            f"Derivatives: {', '.join(str(d) for d in sorted(self.derivatives, key=lambda d: (d.variable, d.wrt, d.order))) or 'none'}",
            f"Max order: {self.max_order}",
            f"Dimension: {self.dimension}",
        ]
        if self.needs_laplacian:
            lines.append(f"Laplacian: {', '.join(sorted(self.needs_laplacian))}")
        if self.needs_gradient:
            lines.append(f"Gradient: {', '.join(sorted(self.needs_gradient))}")
        if self.needs_dalembert:
            lines.append(f"D'Alembertian: {', '.join(sorted(self.needs_dalembert))}")
        return '\n'.join(lines)


# =============================================================================
# JET EXTRACTOR
# =============================================================================

class JetExtractor:
    """Extract derivative requirements from a PDE expression.

    Walks the expression tree and collects all unknowns, coordinates,
    parameters, and derivative requirements.

    Example:
        >>> extractor = JetExtractor()
        >>> parsed = parse("-∇²ψ/2 + x²ψ/2 = Eψ")
        >>> req = extractor.extract(parsed)
        >>> print(req.unknowns)      # {'psi'}
        >>> print(req.needs_laplacian)  # {'psi'}
    """

    # Default parameter values
    DEFAULT_PARAMS = {
        'E': 0.5,
        'hbar': 1.0,
        'm': 1.0,
        'theta': 0.1,
        'omega': 1.0,
        'Lambda': 0.0,
        'alpha': 1.0,
        'beta': 1.0,
        'gamma': 1.0,
        'k': 0.0,
        'c': 1.0,
    }

    # Spatial coordinates (for dimension inference)
    SPATIAL_COORDS = {'x', 'y', 'z', 'r', 'r2'}

    def extract(self, parsed: ParsedPDE) -> JetRequirements:
        """Extract jet requirements from a parsed PDE.

        Args:
            parsed: Parsed PDE equation.

        Returns:
            JetRequirements with all extracted information.
        """
        req = JetRequirements()

        # Walk both sides of the equation
        self._walk(parsed.lhs, req)
        self._walk(parsed.rhs, req)

        # Infer dimension from spatial coordinates
        spatial = self.SPATIAL_COORDS.intersection(req.coordinates)
        req.dimension = max(1, len(spatial))

        return req

    def extract_from_expr(self, expr: PDEExpr) -> JetRequirements:
        """Extract requirements from a single expression."""
        req = JetRequirements()
        self._walk(expr, req)

        spatial = self.SPATIAL_COORDS.intersection(req.coordinates)
        req.dimension = max(1, len(spatial))

        return req

    def _walk(self, expr: PDEExpr, req: JetRequirements):
        """Recursively walk expression tree and extract requirements."""
        if expr is None:
            return

        if isinstance(expr, Variable):
            req.unknowns.add(expr.name)

        elif isinstance(expr, Coordinate):
            req.coordinates.add(expr.name)

        elif isinstance(expr, Parameter):
            default = self.DEFAULT_PARAMS.get(expr.name, expr.default)
            req.parameters[expr.name] = default

        elif isinstance(expr, Constant):
            pass  # No requirements from constants

        elif isinstance(expr, Derivative):
            # Walk inner expression
            self._walk(expr.expr, req)

            # Record derivative requirement
            if isinstance(expr.expr, Variable):
                var_name = expr.expr.name
                if isinstance(expr.wrt, str):
                    req.add_derivative(var_name, expr.wrt, expr.order)
                    req.coordinates.add(expr.wrt)
                else:
                    # Mixed partial - record each
                    for coord in expr.wrt:
                        req.add_derivative(var_name, coord, 1)
                        req.coordinates.add(coord)

        elif isinstance(expr, Laplacian):
            # Walk inner expression
            self._walk(expr.expr, req)

            # Record Laplacian requirement
            if isinstance(expr.expr, Variable):
                req.needs_laplacian.add(expr.expr.name)
                req.max_order = max(req.max_order, 2)

            # Add coordinates if specified
            if expr.coords:
                for coord in expr.coords:
                    req.coordinates.add(coord)

        elif isinstance(expr, Gradient):
            self._walk(expr.expr, req)
            if isinstance(expr.expr, Variable):
                req.needs_gradient.add(expr.expr.name)
                req.max_order = max(req.max_order, 1)

        elif isinstance(expr, Dalembert):
            self._walk(expr.expr, req)
            if isinstance(expr.expr, Variable):
                req.needs_dalembert.add(expr.expr.name)
                req.max_order = max(req.max_order, 2)

            req.coordinates.add(expr.time_coord)
            if expr.spatial_coords:
                for coord in expr.spatial_coords:
                    req.coordinates.add(coord)

        elif isinstance(expr, StarProduct):
            self._walk(expr.left, req)
            self._walk(expr.right, req)
            req.parameters[expr.theta] = self.DEFAULT_PARAMS.get('theta', 0.1)

        elif isinstance(expr, Hamiltonian):
            if expr.args:
                for arg in expr.args:
                    self._walk(arg, req)

        elif isinstance(expr, BinaryOp):
            self._walk(expr.left, req)
            self._walk(expr.right, req)

        elif isinstance(expr, UnaryOp):
            self._walk(expr.operand, req)

        elif isinstance(expr, FunctionCall):
            for arg in expr.args:
                self._walk(arg, req)


# =============================================================================
# EQUATION CLASSIFIER
# =============================================================================

@dataclass
class EquationType:
    """Classification of a PDE type.

    Attributes:
        name: Type name (e.g., 'schrodinger', 'wave').
        category: Broader category (e.g., 'quantum', 'classical').
        is_eigenvalue: Whether this is an eigenvalue problem.
        is_time_dependent: Whether the equation involves time.
        recommended_basis: Suggested basis functions.
        recommended_losses: Suggested loss terms.
    """
    name: str
    category: str
    is_eigenvalue: bool = False
    is_time_dependent: bool = False
    recommended_basis: str = "bspline"
    recommended_losses: List[str] = field(default_factory=list)


class EquationClassifier:
    """Classify PDE type from its structure.

    Uses the jet requirements to determine the equation type
    and provide recommendations for solving it.

    Example:
        >>> classifier = EquationClassifier()
        >>> eq_type = classifier.classify(jet_req)
        >>> print(eq_type.name)  # 'schrodinger'
        >>> print(eq_type.recommended_basis)  # 'hermite'
    """

    def classify(self, req: JetRequirements, parsed: Optional[ParsedPDE] = None) -> EquationType:
        """Classify the equation type from requirements.

        Args:
            req: Jet requirements extracted from the PDE.
            parsed: Optional parsed PDE for additional context.

        Returns:
            EquationType with classification and recommendations.
        """
        # Check for specific patterns

        # Wheeler-DeWitt: involves Ψ or scale factor a
        if 'Psi' in req.unknowns or 'a' in req.coordinates:
            if req.needs_dalembert or 'star' in str(req.derivatives):
                return EquationType(
                    name='deformed_wheeler_dewitt',
                    category='quantum_cosmology',
                    is_eigenvalue=False,
                    recommended_basis='laguerre',
                    recommended_losses=['PDEResidualLoss', 'NormalizationLoss', 'NonTrivialLoss']
                )
            return EquationType(
                name='wheeler_dewitt',
                category='quantum_cosmology',
                is_eigenvalue=False,
                recommended_basis='laguerre',
                recommended_losses=['PDEResidualLoss', 'NormalizationLoss', 'DecayLoss']
            )

        # Schrödinger: Laplacian + E parameter + psi/phi unknown
        if req.needs_laplacian and 'E' in req.parameters:
            if 'psi' in req.unknowns or 'phi' in req.unknowns:
                # Check if harmonic oscillator (x² potential)
                if 'x' in req.coordinates:
                    return EquationType(
                        name='schrodinger',
                        category='quantum',
                        is_eigenvalue=True,
                        recommended_basis='hermite',
                        recommended_losses=['PDEResidualLoss', 'NormalizationLoss', 'BoundaryConditionLoss']
                    )
                return EquationType(
                    name='schrodinger',
                    category='quantum',
                    is_eigenvalue=True,
                    recommended_basis='bspline',
                    recommended_losses=['PDEResidualLoss', 'NormalizationLoss', 'BoundaryConditionLoss']
                )

        # Klein-Gordon: d'Alembertian
        if req.needs_dalembert:
            return EquationType(
                name='klein_gordon',
                category='field_theory',
                is_time_dependent=True,
                recommended_basis='fourier',
                recommended_losses=['PDEResidualLoss', 'BoundaryConditionLoss']
            )

        # Wave equation: ∂²/∂t² and Laplacian
        if 't' in req.coordinates and any(d.wrt == 't' and d.order == 2 for d in req.derivatives):
            if req.needs_laplacian or any(d.order == 2 and d.wrt in ('x', 'y', 'z') for d in req.derivatives):
                return EquationType(
                    name='wave',
                    category='classical',
                    is_time_dependent=True,
                    recommended_basis='fourier',
                    recommended_losses=['PDEResidualLoss', 'BoundaryConditionLoss']
                )

        # Heat equation: ∂/∂t and Laplacian
        if 't' in req.coordinates and any(d.wrt == 't' and d.order == 1 for d in req.derivatives):
            if req.needs_laplacian or any(d.order == 2 and d.wrt in ('x', 'y', 'z') for d in req.derivatives):
                return EquationType(
                    name='heat',
                    category='classical',
                    is_time_dependent=True,
                    recommended_basis='chebyshev',
                    recommended_losses=['PDEResidualLoss', 'BoundaryConditionLoss']
                )

        # Poisson/Laplace: just Laplacian, no time
        if req.needs_laplacian and 't' not in req.coordinates:
            return EquationType(
                name='poisson' if req.unknowns else 'laplace',
                category='classical',
                is_time_dependent=False,
                recommended_basis='chebyshev',
                recommended_losses=['PDEResidualLoss', 'BoundaryConditionLoss']
            )

        # Wigner function: r² coordinate, W unknown
        if 'W' in req.unknowns or 'r2' in req.coordinates:
            return EquationType(
                name='wigner',
                category='quantum',
                is_eigenvalue='E' in req.parameters,
                recommended_basis='laguerre',
                recommended_losses=['PDEResidualLoss', 'NormalizationLoss', 'NonTrivialLoss', 'EigenvalueLoss']
            )

        # Generic fallback
        return EquationType(
            name='generic',
            category='unknown',
            is_eigenvalue='E' in req.parameters,
            is_time_dependent='t' in req.coordinates,
            recommended_basis='bspline',
            recommended_losses=['PDEResidualLoss', 'BoundaryConditionLoss']
        )


# =============================================================================
# FULL ANALYSIS
# =============================================================================

@dataclass
class PDEAnalysis:
    """Complete analysis of a parsed PDE.

    Combines jet requirements, equation classification, and
    additional metadata about the equation.

    Attributes:
        parsed: The original parsed PDE.
        jet_req: Extracted jet requirements.
        equation_type: Classified equation type.
        unknowns: Set of unknown function names.
        coordinates: Set of coordinate names.
        parameters: Dict of parameter names to defaults.
        dimension: Spatial dimension.
    """
    parsed: ParsedPDE
    jet_req: JetRequirements
    equation_type: EquationType

    @property
    def unknowns(self) -> Set[str]:
        return self.jet_req.unknowns

    @property
    def coordinates(self) -> Set[str]:
        return self.jet_req.coordinates

    @property
    def parameters(self) -> Dict[str, Optional[float]]:
        return self.jet_req.parameters

    @property
    def dimension(self) -> int:
        return self.jet_req.dimension

    @property
    def is_eigenvalue_problem(self) -> bool:
        return self.equation_type.is_eigenvalue

    @property
    def is_time_dependent(self) -> bool:
        return self.equation_type.is_time_dependent

    @property
    def recommended_basis(self) -> str:
        return self.equation_type.recommended_basis

    @property
    def recommended_losses(self) -> List[str]:
        return self.equation_type.recommended_losses

    def summary(self) -> str:
        """Get a human-readable summary of the analysis."""
        lines = [
            f"Equation: {self.parsed.original}",
            f"Type: {self.equation_type.name} ({self.equation_type.category})",
            f"Eigenvalue problem: {self.is_eigenvalue_problem}",
            f"Time-dependent: {self.is_time_dependent}",
            f"Dimension: {self.dimension}",
            "",
            "Jet Requirements:",
            self.jet_req.summary(),
            "",
            f"Recommended basis: {self.recommended_basis}",
            f"Recommended losses: {', '.join(self.recommended_losses)}",
        ]
        return '\n'.join(lines)


def analyze(parsed: ParsedPDE) -> PDEAnalysis:
    """Perform complete analysis of a parsed PDE.

    Args:
        parsed: Parsed PDE from the parser.

    Returns:
        PDEAnalysis with all extracted information.

    Example:
        >>> from kan_mlx_physics.pde.parser import parse
        >>> from kan_mlx_physics.pde.analysis import analyze

        >>> parsed = parse("-∇²ψ/2 + x²ψ/2 = Eψ")
        >>> analysis = analyze(parsed)
        >>> print(analysis.summary())
    """
    extractor = JetExtractor()
    jet_req = extractor.extract(parsed)

    classifier = EquationClassifier()
    equation_type = classifier.classify(jet_req, parsed)

    return PDEAnalysis(
        parsed=parsed,
        jet_req=jet_req,
        equation_type=equation_type
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_unknowns(parsed: ParsedPDE) -> Set[str]:
    """Extract unknown function names from a parsed PDE."""
    extractor = JetExtractor()
    req = extractor.extract(parsed)
    return req.unknowns


def get_coordinates(parsed: ParsedPDE) -> Set[str]:
    """Extract coordinate names from a parsed PDE."""
    extractor = JetExtractor()
    req = extractor.extract(parsed)
    return req.coordinates


def get_parameters(parsed: ParsedPDE) -> Dict[str, Optional[float]]:
    """Extract parameters with defaults from a parsed PDE."""
    extractor = JetExtractor()
    req = extractor.extract(parsed)
    return req.parameters


def get_derivative_orders(parsed: ParsedPDE) -> Dict[str, int]:
    """Get maximum derivative order for each coordinate."""
    extractor = JetExtractor()
    req = extractor.extract(parsed)

    orders = {}
    for deriv in req.derivatives:
        wrt = deriv.wrt
        if wrt not in orders or deriv.order > orders[wrt]:
            orders[wrt] = deriv.order

    return orders


def infer_dimension(parsed: ParsedPDE) -> int:
    """Infer spatial dimension from a parsed PDE."""
    extractor = JetExtractor()
    req = extractor.extract(parsed)
    return req.dimension
