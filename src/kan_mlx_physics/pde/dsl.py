"""Physics DSL: Type PDEs naturally, solve them with KANs.

Write equations as you think them:

    ψ = solve("−ℏ²∇²ψ/2m + V(x)ψ = Eψ", domain=[-5, 5])

    Ψ = solve("Ĥ Ψ = 0", problem="wheeler-dewitt")

    φ = solve("□φ − m²φ = 0", spacetime=[(0,10), (-5,5)])

Physics first. Always.
"""

import mlx.core as mx
import numpy as np
import re
from typing import Optional, Dict, List, Tuple, Callable, Union
from dataclasses import dataclass, field

from .problem import PDEProblem, Domain, BoundaryCondition
from .operators import laplacian, grad, _fd_derivative, _fd_second_derivative
from .physics import (
    WheelerDeWitt, Friedmann, KleinGordonCurved,
    Schrodinger, DeformedSchrodinger, DeformedWheelerDeWitt
)
from .solver import solve as _solve, SolverConfig


# =============================================================================
# EQUATION PARSER
# =============================================================================

@dataclass
class ParsedEquation:
    """Parsed PDE structure."""
    lhs: str                          # Left-hand side
    rhs: str                          # Right-hand side
    unknown: str                      # Unknown function (ψ, φ, Ψ, etc.)
    operators: List[str]              # Detected operators (∇², ∂/∂t, etc.)
    parameters: Dict[str, float]      # Extracted parameters
    equation_type: str                # "schrodinger", "wave", "wheeler-dewitt", etc.


class EquationParser:
    """Parse physics equations from natural notation."""

    # Operator patterns
    OPERATORS = {
        # Laplacian
        r'∇²|\\nabla\^2|laplacian|Δ': 'laplacian',
        r'∇|\\nabla|grad': 'gradient',
        # Time derivatives
        r'∂²/∂t²|∂_t²|∂tt': 'd2_dt2',
        r'∂/∂t|∂_t|∂t': 'd_dt',
        # Spatial derivatives
        r'∂²/∂x²|∂_x²|∂xx': 'd2_dx2',
        r'∂/∂x|∂_x|∂x': 'd_dx',
        # D'Alembertian
        r'□|\\square|\\Box|dalembert': 'dalembert',
        # Star product
        r'⋆|\\star|★': 'star',
        # Hamiltonian
        r'Ĥ|\\hat\{H\}|H\^': 'hamiltonian',
    }

    # Unknown function patterns
    UNKNOWNS = {
        r'ψ|\\psi|psi': 'psi',
        r'Ψ|\\Psi|PSI': 'Psi',
        r'φ|\\phi|phi': 'phi',
        r'Φ|\\Phi|PHI': 'Phi',
        r'u': 'u',
        r'v': 'v',
        r'f': 'f',
    }

    # Parameter patterns
    PARAMETERS = {
        r'ℏ|\\hbar|hbar': ('hbar', 1.0),
        r'm(?![a-z])': ('mass', 1.0),
        r'E(?![a-z])': ('energy', 0.5),
        r'ω|\\omega|omega': ('omega', 1.0),
        r'λ|\\lambda|lambda|Λ|\\Lambda': ('lambda', 0.0),
        r'θ|\\theta|theta': ('theta', 0.1),
        r'k(?![a-z])': ('k', 0.0),
    }

    # Equation type signatures
    SIGNATURES = {
        'schrodinger': ['laplacian', 'psi', 'energy'],
        'wheeler-dewitt': ['hamiltonian', 'Psi'],
        'klein-gordon': ['dalembert', 'phi'],
        'wave': ['d2_dt2', 'laplacian'],
        'heat': ['d_dt', 'laplacian'],
        'poisson': ['laplacian'],
    }

    def parse(self, equation: str) -> ParsedEquation:
        """Parse an equation string."""
        # Normalize
        eq = equation.strip()

        # Split on =
        if '=' in eq:
            parts = eq.split('=')
            lhs, rhs = parts[0].strip(), parts[1].strip()
        else:
            lhs, rhs = eq, '0'

        # Detect unknown
        unknown = 'psi'
        for pattern, name in self.UNKNOWNS.items():
            if re.search(pattern, eq):
                unknown = name
                break

        # Detect operators
        operators = []
        for pattern, name in self.OPERATORS.items():
            if re.search(pattern, eq, re.IGNORECASE):
                operators.append(name)

        # Extract parameters
        parameters = {}
        for pattern, (name, default) in self.PARAMETERS.items():
            if re.search(pattern, eq):
                parameters[name] = default

        # Determine equation type
        equation_type = self._classify(operators, unknown)

        return ParsedEquation(
            lhs=lhs,
            rhs=rhs,
            unknown=unknown,
            operators=operators,
            parameters=parameters,
            equation_type=equation_type
        )

    def _classify(self, operators: List[str], unknown: str) -> str:
        """Classify equation type from operators."""
        op_set = set(operators)

        if 'hamiltonian' in op_set and unknown == 'Psi':
            return 'wheeler-dewitt'
        if 'dalembert' in op_set:
            return 'klein-gordon'
        if 'laplacian' in op_set:
            if 'd_dt' in op_set:
                return 'heat'
            if 'd2_dt2' in op_set:
                return 'wave'
            return 'schrodinger'
        if 'star' in op_set:
            return 'deformed'

        return 'generic'


# =============================================================================
# EQUATION TEMPLATES
# =============================================================================

TEMPLATES = {
    # Quantum Mechanics
    "schrodinger": "−ℏ²∇²ψ/2m + V(x)ψ = Eψ",
    "harmonic": "−ℏ²∇²ψ/2m + ½mω²x²ψ = Eψ",
    "hydrogen": "−ℏ²∇²ψ/2m − e²/r ψ = Eψ",
    "particle-box": "−ℏ²∇²ψ/2m = Eψ",

    # Quantum Cosmology
    "wheeler-dewitt": "Ĥ Ψ = 0",
    "wheeler-dewitt-full": "[−ℏ² ∂²/∂a² + U(a)] Ψ = 0",
    "deformed-wdw": "Ĥ ⋆_θ Ψ = 0",

    # Field Theory
    "klein-gordon": "□φ − m²φ = 0",
    "klein-gordon-curved": "□φ − m²φ − ξRφ = 0",
    "wave": "∂²u/∂t² = c²∇²u",

    # Cosmology
    "friedmann": "H² = 8πGρ/3 − k/a² + Λ/3",
    "mukhanov-sasaki": "v'' + (k² − z''/z)v = 0",

    # Classical
    "heat": "∂u/∂t = α∇²u",
    "laplace": "∇²u = 0",
    "poisson": "∇²u = f",
}


# =============================================================================
# MAIN DSL INTERFACE
# =============================================================================

def equation(eq: str) -> ParsedEquation:
    """Parse a physics equation.

    Args:
        eq: Equation string in natural notation

    Returns:
        ParsedEquation object

    Example:
        >>> equation("−∇²ψ + x²ψ = Eψ")
        ParsedEquation(type='schrodinger', unknown='psi', ...)
    """
    parser = EquationParser()
    return parser.parse(eq)


def solve(
    eq: str,
    domain: Optional[Union[List, Tuple, Domain]] = None,
    params: Optional[Dict[str, float]] = None,
    bc: Optional[str] = None,
    steps: int = 500,
    verbose: bool = True,
    **kwargs
) -> Tuple["MultKAN", Dict]:
    """Solve a PDE from its equation.

    The main interface. Type your equation, get a solution.

    Args:
        eq: Equation string or template name
        domain: Domain specification
        params: Parameter values (E, ℏ, m, θ, etc.)
        bc: Boundary conditions ("dirichlet", "periodic", "cauchy")
        steps: Training steps
        verbose: Print progress
        **kwargs: Additional solver options

    Returns:
        (model, history) tuple

    Examples:
        # Harmonic oscillator ground state
        ψ, _ = solve("−∇²ψ/2 + x²ψ/2 = Eψ", domain=[-5, 5], params={"E": 0.5})

        # Wheeler-DeWitt with deformation
        Ψ, _ = solve("Ĥ ⋆_θ Ψ = 0", domain=[0.1, 5], params={"θ": 0.05})

        # Klein-Gordon in de Sitter
        φ, _ = solve("□φ = 0", domain={"t": [0, 5], "x": [-5, 5]})

        # Use template
        ψ, _ = solve("schrodinger", domain=[-5, 5])
    """
    # Check if eq is a template name
    if eq.lower() in TEMPLATES:
        eq = TEMPLATES[eq.lower()]

    # Parse equation
    parsed = equation(eq)

    # Merge parameters
    parameters = parsed.parameters.copy()
    if params:
        parameters.update(params)

    # Build domain
    if domain is None:
        domain = Domain.interval(-5, 5)
    elif isinstance(domain, (list, tuple)):
        if len(domain) == 2 and isinstance(domain[0], (int, float)):
            domain = Domain.interval(domain[0], domain[1])
        else:
            domain = Domain([(d[0], d[1]) for d in domain])
    elif isinstance(domain, dict):
        bounds = list(domain.values())
        domain = Domain(bounds)

    # Map to physics problem
    problem = _build_problem(parsed, domain, parameters, bc)

    # Configure solver
    config = SolverConfig(
        steps=steps,
        verbose=verbose,
        width=[domain.dim, 20, 20, 1],
        **{k: v for k, v in kwargs.items() if hasattr(SolverConfig, k)}
    )

    # Solve
    return _solve(problem, config)


def _build_problem(
    parsed: ParsedEquation,
    domain: Domain,
    parameters: Dict[str, float],
    bc: Optional[str]
) -> PDEProblem:
    """Build PDEProblem from parsed equation."""

    eq_type = parsed.equation_type

    if eq_type == 'schrodinger':
        E = parameters.get('energy', 0.5)
        hbar = parameters.get('hbar', 1.0)
        m = parameters.get('mass', 1.0)
        return Schrodinger(energy=E, hbar=hbar, mass=m, domain=domain)

    elif eq_type == 'wheeler-dewitt':
        hbar = parameters.get('hbar', 1.0)
        return WheelerDeWitt(hbar=hbar, domain=domain)

    elif eq_type == 'deformed':
        theta = parameters.get('theta', 0.1)
        hbar = parameters.get('hbar', 1.0)
        if parsed.unknown == 'Psi':
            return DeformedWheelerDeWitt(hbar=hbar, theta=theta, domain=domain)
        else:
            E = parameters.get('energy', 0.5)
            return DeformedSchrodinger(energy=E, hbar=hbar, theta=theta, domain=domain)

    elif eq_type == 'klein-gordon':
        m = parameters.get('mass', 0.0)
        return KleinGordonCurved(mass=m, domain=domain)

    elif eq_type == 'heat' or eq_type == 'wave':
        return _generic_pde(parsed, domain, parameters)

    else:
        return _generic_pde(parsed, domain, parameters)


def _generic_pde(
    parsed: ParsedEquation,
    domain: Domain,
    parameters: Dict[str, float]
) -> PDEProblem:
    """Build generic PDE from parsed equation."""

    def residual(u: Callable, x: mx.array) -> mx.array:
        """Generic residual based on detected operators."""
        result = mx.zeros(x.shape[0])

        u_val = u(x)
        if len(u_val.shape) > 1:
            u_val = u_val[:, 0]

        for op in parsed.operators:
            if op == 'laplacian':
                result = result + laplacian(u, x)
            elif op == 'd2_dt2':
                result = result + _fd_second_derivative(u, x, 0)
            elif op == 'd_dt':
                result = result + _fd_derivative(u, x, 0)

        return result

    return PDEProblem(
        name=f"Generic: {parsed.lhs} = {parsed.rhs}",
        domain=domain,
        residual=residual,
        parameters=parameters
    )


# =============================================================================
# CONVENIENT SHORTCUTS
# =============================================================================

def schrodinger(V: Optional[Callable] = None, E: float = 0.5, domain=None):
    """Shortcut for Schrödinger equation."""
    return solve("schrodinger", domain=domain or [-5, 5], params={"E": E})


def wheeler_dewitt(theta: float = 0.0, domain=None):
    """Shortcut for Wheeler-DeWitt equation."""
    if theta > 0:
        return solve("Ĥ ⋆_θ Ψ = 0", domain=domain or [0.1, 5], params={"θ": theta})
    return solve("wheeler-dewitt", domain=domain or [0.1, 5])


def klein_gordon(m: float = 0.0, domain=None):
    """Shortcut for Klein-Gordon equation."""
    return solve("klein-gordon", domain=domain or {"t": [0, 5], "x": [-5, 5]}, params={"m": m})


def wave(c: float = 1.0, domain=None):
    """Shortcut for wave equation."""
    return solve("wave", domain=domain or {"t": [0, 5], "x": [-5, 5]})


# =============================================================================
# LIST AVAILABLE EQUATIONS
# =============================================================================

def list_equations() -> Dict[str, str]:
    """List all available equation templates."""
    return TEMPLATES.copy()


def help_equation(name: str) -> str:
    """Get help for a specific equation template."""
    if name not in TEMPLATES:
        return f"Unknown equation: {name}. Use list_equations() to see available templates."

    eq = TEMPLATES[name]
    parsed = equation(eq)

    help_text = f"""
{name}
{'='*len(name)}

Equation: {eq}

Type: {parsed.equation_type}
Unknown: {parsed.unknown}
Operators: {', '.join(parsed.operators) or 'none detected'}
Parameters: {parsed.parameters or 'none'}

Usage:
    model, history = solve("{name}", domain=[...], params={{...}})
"""
    return help_text
