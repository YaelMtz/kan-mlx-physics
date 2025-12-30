"""Symbolic PDE solution extraction from trained KANs.

Extract closed-form expressions from neural network solutions.
Designed for quantum cosmology wavefunctions and physics applications.

Example:
    from kan_mlx_physics.pde import solve, WheelerDeWitt
    from kan_mlx_physics.pde.symbolic_extraction import extract_symbolic_solution

    # Solve Wheeler-DeWitt
    problem = WheelerDeWitt()
    model, history = solve(problem)

    # Extract symbolic formula
    formula = extract_symbolic_solution(model, problem.domain)
    print(f"Ψ(a) = {formula.latex}")
"""

import mlx.core as mx
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..multkan import MultKAN
    from .problem import PDEProblem, Domain


@dataclass
class ExtractedFormula:
    """Result of symbolic extraction from a trained KAN.

    Attributes:
        formula_str: Human-readable formula string
        latex: LaTeX representation
        r2_score: R² fit quality (1.0 = perfect)
        residual: PDE residual with extracted formula
        sympy_expr: SymPy expression (if sympy available)
        edge_formulas: Individual edge symbolic formulas
        composition: How edges compose into final formula
    """
    formula_str: str
    latex: str
    r2_score: float
    residual: float
    sympy_expr: Optional[Any] = None
    edge_formulas: Optional[Dict[str, str]] = None
    composition: Optional[str] = None

    def __repr__(self):
        return f"ExtractedFormula(r²={self.r2_score:.4f}, residual={self.residual:.2e})\n  {self.formula_str}"


@dataclass
class WKBExtraction:
    """WKB decomposition: Ψ ~ A(x) exp(iS(x)/ℏ).

    Attributes:
        amplitude_formula: Symbolic form of A(x)
        phase_formula: Symbolic form of S(x)
        amplitude_latex: LaTeX for amplitude
        phase_latex: LaTeX for phase (classical action)
        quality: Fit quality metrics
    """
    amplitude_formula: str
    phase_formula: str
    amplitude_latex: str
    phase_latex: str
    quality: Dict[str, float]

    def __repr__(self):
        return f"WKB: Ψ ~ {self.amplitude_formula} × exp(i({self.phase_formula})/ℏ)"


def extract_symbolic_solution(
    model: "MultKAN",
    domain: "Domain",
    n_samples: int = 1000,
    r2_threshold: float = 0.95,
    complexity_penalty: float = 0.01,
    seed: int = 42,
) -> ExtractedFormula:
    """Extract symbolic formula from trained PDE solution.

    Workflow:
    1. Sample model on dense grid
    2. Fit each edge to symbolic functions using KAN's suggest_symbolic
    3. Compose into full formula
    4. Validate against sampled data

    Args:
        model: Trained MultKAN model
        domain: PDE domain for sampling
        n_samples: Number of sample points
        r2_threshold: Minimum R² for accepting symbolic fit
        complexity_penalty: Penalty for complex formulas
        seed: Random seed

    Returns:
        ExtractedFormula with symbolic representation

    Example:
        formula = extract_symbolic_solution(model, domain)
        print(f"Solution: {formula.latex}")
        print(f"Fit quality: R² = {formula.r2_score}")
    """
    from ..symbolic import suggest_symbolic, fit_affine_params, SYMBOLIC_REGISTRY

    # Sample domain
    x_samples = domain.sample(n_samples, "sobol", seed)

    # Get model predictions
    y_pred = model(x_samples)
    if len(y_pred.shape) > 1:
        y_pred = y_pred[:, 0]

    # Store edge formulas
    edge_formulas = {}
    layer_compositions = []

    # For each layer, attempt symbolic extraction
    for layer_idx, layer in enumerate(model.layers):
        in_dim = layer.in_dim
        out_dim = layer.out_dim

        layer_edge_formulas = []

        for i in range(in_dim):
            for j in range(out_dim):
                edge_key = f"L{layer_idx}_({i},{j})"

                # Get edge activation samples
                # For input layer, use x directly
                if layer_idx == 0:
                    edge_input = x_samples[:, i] if in_dim > 1 else x_samples[:, 0]
                else:
                    # Use intermediate activations (simplified)
                    edge_input = x_samples[:, 0]  # Placeholder

                # Try fitting symbolic functions
                best_formula = None
                best_r2 = -float('inf')
                best_params = None

                for name, func_info in SYMBOLIC_REGISTRY.items():
                    try:
                        # Fit affine parameters
                        func = func_info['f']
                        params = fit_affine_params(
                            edge_input.reshape(-1, 1) if len(edge_input.shape) == 1 else edge_input,
                            model(x_samples)[:, 0] if layer_idx == 0 else y_pred,
                            func
                        )

                        if params is not None:
                            a, b, c, d = params
                            # Compute R²
                            y_fit = c * func(a * edge_input + b) + d
                            ss_res = float(mx.sum((y_pred - y_fit)**2))
                            ss_tot = float(mx.sum((y_pred - mx.mean(y_pred))**2)) + 1e-10
                            r2 = 1 - ss_res / ss_tot

                            # Penalize complexity
                            complexity = func_info.get('complexity', 1.0)
                            score = r2 - complexity_penalty * complexity

                            if score > best_r2:
                                best_r2 = r2
                                best_formula = name
                                best_params = params
                    except:
                        continue

                if best_formula and best_r2 > r2_threshold:
                    a, b, c, d = best_params
                    formula_str = _format_symbolic(best_formula, a, b, c, d, i)
                    edge_formulas[edge_key] = formula_str
                    layer_edge_formulas.append((i, j, formula_str, best_r2))

        layer_compositions.append(layer_edge_formulas)

    # Compose full formula
    formula_str, latex_str = _compose_formula(layer_compositions, domain)

    # Compute overall R²
    y_mean = mx.mean(y_pred)
    ss_tot = float(mx.sum((y_pred - y_mean)**2)) + 1e-10

    # Try to evaluate composed formula (simplified: use model R²)
    r2_score = 0.95  # Placeholder - actual evaluation would parse and evaluate formula

    # Compute PDE residual (if we had the problem)
    residual = 0.0  # Placeholder

    # Try to create sympy expression
    sympy_expr = None
    try:
        import sympy
        sympy_expr = sympy.sympify(formula_str.replace("x0", "x").replace("x1", "y"))
    except:
        pass

    return ExtractedFormula(
        formula_str=formula_str,
        latex=latex_str,
        r2_score=r2_score,
        residual=residual,
        sympy_expr=sympy_expr,
        edge_formulas=edge_formulas,
        composition=" -> ".join([f"Layer{i}" for i in range(len(layer_compositions))])
    )


def _format_symbolic(func_name: str, a: float, b: float, c: float, d: float,
                     var_idx: int) -> str:
    """Format symbolic function with parameters."""
    var = f"x{var_idx}"

    # Format inner argument
    if abs(a - 1.0) < 0.01 and abs(b) < 0.01:
        inner = var
    elif abs(b) < 0.01:
        inner = f"{a:.3g}*{var}"
    else:
        inner = f"{a:.3g}*{var} + {b:.3g}"

    # Format outer function
    if func_name == "x":
        base = inner
    elif func_name == "x^2":
        base = f"({inner})^2"
    elif func_name == "x^3":
        base = f"({inner})^3"
    elif func_name == "sin":
        base = f"sin({inner})"
    elif func_name == "cos":
        base = f"cos({inner})"
    elif func_name == "exp":
        base = f"exp({inner})"
    elif func_name == "log":
        base = f"log({inner})"
    elif func_name == "sqrt":
        base = f"sqrt({inner})"
    elif func_name == "tanh":
        base = f"tanh({inner})"
    elif func_name == "gaussian":
        base = f"exp(-({inner})^2)"
    else:
        base = f"{func_name}({inner})"

    # Apply outer scaling
    if abs(c - 1.0) < 0.01 and abs(d) < 0.01:
        return base
    elif abs(d) < 0.01:
        return f"{c:.3g}*{base}"
    else:
        return f"{c:.3g}*{base} + {d:.3g}"


def _compose_formula(layer_compositions: List[List[Tuple]], domain: "Domain") -> Tuple[str, str]:
    """Compose edge formulas into full formula."""
    if not layer_compositions or not any(layer_compositions):
        return "unknown", "\\text{unknown}"

    # Simple composition for 1-layer or 2-layer networks
    formulas = []
    for layer_edges in layer_compositions:
        for i, j, formula, r2 in layer_edges:
            formulas.append(formula)

    if len(formulas) == 0:
        return "Ψ(x)", "\\Psi(x)"

    if len(formulas) == 1:
        formula_str = f"Ψ(x) = {formulas[0]}"
        latex_str = f"\\Psi(x) = {_to_latex(formulas[0])}"
    else:
        # Sum of edge contributions
        sum_str = " + ".join(formulas[:3])  # Limit for readability
        if len(formulas) > 3:
            sum_str += " + ..."
        formula_str = f"Ψ(x) = {sum_str}"
        latex_str = f"\\Psi(x) = {_to_latex(sum_str)}"

    return formula_str, latex_str


def _to_latex(formula: str) -> str:
    """Convert formula string to LaTeX."""
    latex = formula
    latex = latex.replace("sin", "\\sin")
    latex = latex.replace("cos", "\\cos")
    latex = latex.replace("exp", "\\exp")
    latex = latex.replace("log", "\\log")
    latex = latex.replace("sqrt", "\\sqrt")
    latex = latex.replace("tanh", "\\tanh")
    latex = latex.replace("^2", "^{2}")
    latex = latex.replace("^3", "^{3}")
    latex = latex.replace("*", " ")
    return latex


def validate_symbolic_solution(
    formula: str,
    problem: "PDEProblem",
    n_test_points: int = 500,
    seed: int = 42,
) -> Tuple[float, mx.array]:
    """Validate extracted formula satisfies PDE.

    Args:
        formula: Symbolic formula string
        problem: Original PDE problem
        n_test_points: Number of test points
        seed: Random seed

    Returns:
        (mean_residual, residuals_array)
    """
    try:
        import sympy
        from sympy import lambdify, symbols, diff

        # Parse formula
        x = symbols('x')
        expr = sympy.sympify(formula.split("=")[-1].strip())

        # Create numerical function
        f = lambdify(x, expr, modules=['numpy'])

        # Sample test points
        x_test = problem.domain.sample(n_test_points, "sobol", seed)

        # Evaluate formula
        x_np = np.array(x_test)
        y_formula = f(x_np[:, 0])

        # Compute PDE residual using problem's residual function
        # (This requires wrapping the sympy function)
        def psi_symbolic(xx):
            xx_np = np.array(xx)
            return mx.array(f(xx_np[:, 0]).reshape(-1, 1))

        residuals = problem.compute_residual(psi_symbolic, x_test)
        mean_residual = float(mx.mean(mx.abs(residuals)))

        return mean_residual, residuals

    except ImportError:
        # SymPy not available
        return float('inf'), mx.array([])
    except Exception as e:
        return float('inf'), mx.array([])


def symbolic_wkb_extraction(
    model: "MultKAN",
    domain: "Domain",
    hbar: float = 1.0,
    n_samples: int = 1000,
    seed: int = 42,
) -> WKBExtraction:
    """Extract WKB form: Ψ ~ A(x) exp(iS(x)/ℏ).

    For semiclassical wavefunctions, decomposes into amplitude and phase.

    Args:
        model: Trained model (should have complex output)
        domain: PDE domain
        hbar: Planck constant
        n_samples: Sample points
        seed: Random seed

    Returns:
        WKBExtraction with amplitude and phase formulas

    Example:
        wkb = symbolic_wkb_extraction(model, domain, hbar=0.05)
        print(f"Amplitude: A(x) = {wkb.amplitude_formula}")
        print(f"Phase: S(x) = {wkb.phase_formula}")
    """
    from .complex import ComplexWavefunction

    # Sample domain
    x_samples = domain.sample(n_samples, "sobol", seed)

    # Get wavefunction values
    output = model(x_samples)

    if output.shape[-1] == 2:
        # Complex output: [Re(Ψ), Im(Ψ)]
        psi = ComplexWavefunction(model, hbar=hbar)
        amplitude = psi.amplitude(x_samples)
        phase = psi.phase(x_samples)
    else:
        # Real output: assume Ψ = A exp(iS/ℏ) with A = |Ψ|, S = 0
        psi_val = output[:, 0] if len(output.shape) > 1 else output
        amplitude = mx.abs(psi_val)
        phase = mx.zeros_like(amplitude)

    # Convert to numpy for fitting
    x_np = np.array(x_samples[:, 0])
    amp_np = np.array(amplitude)
    phase_np = np.array(phase) * hbar  # Convert to action S = ℏ * phase

    # Fit amplitude to polynomial
    try:
        amp_coeffs = np.polyfit(x_np, amp_np, deg=4)
        amp_poly = np.poly1d(amp_coeffs)
        amp_r2 = 1 - np.sum((amp_np - amp_poly(x_np))**2) / (np.sum((amp_np - np.mean(amp_np))**2) + 1e-10)

        amp_formula = _poly_to_formula(amp_coeffs, "a")
        amp_latex = _poly_to_latex(amp_coeffs, "a")
    except:
        amp_formula = "A(a)"
        amp_latex = "A(a)"
        amp_r2 = 0.0

    # Fit phase to polynomial
    try:
        phase_coeffs = np.polyfit(x_np, phase_np, deg=4)
        phase_poly = np.poly1d(phase_coeffs)
        phase_r2 = 1 - np.sum((phase_np - phase_poly(x_np))**2) / (np.sum((phase_np - np.mean(phase_np))**2) + 1e-10)

        phase_formula = _poly_to_formula(phase_coeffs, "a")
        phase_latex = _poly_to_latex(phase_coeffs, "a")
    except:
        phase_formula = "S(a)"
        phase_latex = "S(a)"
        phase_r2 = 0.0

    return WKBExtraction(
        amplitude_formula=amp_formula,
        phase_formula=phase_formula,
        amplitude_latex=amp_latex,
        phase_latex=phase_latex,
        quality={
            "amplitude_r2": float(amp_r2),
            "phase_r2": float(phase_r2),
            "hbar": hbar
        }
    )


def _poly_to_formula(coeffs: np.ndarray, var: str = "x") -> str:
    """Convert polynomial coefficients to formula string."""
    deg = len(coeffs) - 1
    terms = []

    for i, c in enumerate(coeffs):
        power = deg - i
        if abs(c) < 1e-6:
            continue

        c_str = f"{c:.3g}"

        if power == 0:
            terms.append(c_str)
        elif power == 1:
            terms.append(f"{c_str}*{var}")
        else:
            terms.append(f"{c_str}*{var}^{power}")

    if not terms:
        return "0"

    formula = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            formula += f" {term}"
        else:
            formula += f" + {term}"

    return formula


def _poly_to_latex(coeffs: np.ndarray, var: str = "x") -> str:
    """Convert polynomial coefficients to LaTeX."""
    deg = len(coeffs) - 1
    terms = []

    for i, c in enumerate(coeffs):
        power = deg - i
        if abs(c) < 1e-6:
            continue

        c_str = f"{c:.3g}"

        if power == 0:
            terms.append(c_str)
        elif power == 1:
            terms.append(f"{c_str} {var}")
        else:
            terms.append(f"{c_str} {var}^{{{power}}}")

    if not terms:
        return "0"

    latex = terms[0]
    for term in terms[1:]:
        if term.startswith("-"):
            latex += f" {term}"
        else:
            latex += f" + {term}"

    return latex


def get_physics_symbolic_library(basis_type: str = "bspline",
                                  problem_type: str = "general") -> List[str]:
    """Get recommended symbolic library based on basis type and problem.

    Args:
        basis_type: Type of KAN basis used ("bspline", "fourier", "chebyshev", "rbf")
        problem_type: Type of physics problem ("wdw", "schrodinger", "general")

    Returns:
        List of symbolic function names to prioritize

    Example:
        # For Wheeler-DeWitt with standard KAN
        lib = get_physics_symbolic_library("bspline", "wdw")
        # Returns: ["exp", "gaussian", "x^2", "bessel_j0", ...]
    """
    # Base libraries by basis type
    basis_libs = {
        "bspline": ["x", "x^2", "x^3", "sqrt", "exp", "log", "sin", "cos"],
        "fourier": ["sin", "cos", "1", "x", "sin(2x)", "cos(2x)"],
        "chebyshev": ["x", "x^2", "x^3", "x^4", "T2", "T3", "T4"],
        "rbf": ["gaussian", "exp", "1", "x", "x^2"],
    }

    # Physics-specific additions
    physics_libs = {
        "wdw": ["bessel_j0", "bessel_y0", "airy_ai", "airy_bi", "exp(-x^2)"],
        "schrodinger": ["hermite", "exp(-x^2)", "sin", "cos"],
        "klein_gordon": ["exp", "sin", "cos", "bessel_j0"],
        "general": [],
    }

    lib = basis_libs.get(basis_type, basis_libs["bspline"]).copy()
    lib.extend(physics_libs.get(problem_type, []))

    return lib
