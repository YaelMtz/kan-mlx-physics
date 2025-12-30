"""Beautiful formula rendering for KAN-MLX-Physics.

This module provides multiple output formats for symbolic formulas:
- Unicode (terminal-friendly with mathematical symbols)
- LaTeX (for papers and Jupyter)
- Typst (for modern typesetting)
- SymPy (for computer algebra manipulation)
- ASCII (plain text fallback)

Example:
    from kan_mlx_physics import MultKAN
    from kan_mlx_physics.formula_render import FormulaRenderer, render_formula

    model = MultKAN(width=[1, 2, 1], grid=5)
    # ... train and auto_symbolic ...

    # Quick render
    print(render_formula(model, format="unicode"))

    # Or use the renderer for more control
    renderer = FormulaRenderer(model)
    print(renderer.unicode())
    print(renderer.latex())
    renderer.display()  # Rich display in Jupyter
"""

import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

# Unicode mathematical symbols
UNICODE_SYMBOLS = {
    # Greek letters
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω",

    # Operators and relations
    "times": "×", "cdot": "·", "div": "÷",
    "pm": "±", "mp": "∓",
    "leq": "≤", "geq": "≥", "neq": "≠",
    "approx": "≈", "equiv": "≡", "propto": "∝",
    "infty": "∞", "nabla": "∇", "partial": "∂",
    "sum": "∑", "prod": "∏", "int": "∫",
    "sqrt": "√",

    # Subscripts and superscripts
    "^0": "⁰", "^1": "¹", "^2": "²", "^3": "³", "^4": "⁴",
    "^5": "⁵", "^6": "⁶", "^7": "⁷", "^8": "⁸", "^9": "⁹",
    "^+": "⁺", "^-": "⁻", "^n": "ⁿ", "^i": "ⁱ",
    "_0": "₀", "_1": "₁", "_2": "₂", "_3": "₃", "_4": "₄",
    "_5": "₅", "_6": "₆", "_7": "₇", "_8": "₈", "_9": "₉",
    "_+": "₊", "_-": "₋", "_a": "ₐ", "_e": "ₑ",
    "_i": "ᵢ", "_j": "ⱼ", "_k": "ₖ", "_n": "ₙ", "_x": "ₓ",

    # Arrows
    "to": "→", "from": "←", "iff": "↔",
    "implies": "⟹", "iff": "⟺",

    # Brackets
    "langle": "⟨", "rangle": "⟩",
    "lceil": "⌈", "rceil": "⌉",
    "lfloor": "⌊", "rfloor": "⌋",
}

# Function name to Unicode mapping
UNICODE_FUNCTIONS = {
    "sin": "sin", "cos": "cos", "tan": "tan",
    "sinh": "sinh", "cosh": "cosh", "tanh": "tanh",
    "arcsin": "arcsin", "arccos": "arccos", "arctan": "arctan",
    "exp": "exp", "log": "ln", "ln": "ln",
    "sqrt": "√", "abs": "|·|",
    "sign": "sgn", "sgn": "sgn",
    "sigmoid": "σ",
    "gaussian": "exp(-x²)",
    "erf": "erf", "erfc": "erfc",
    "gamma": "Γ", "Gamma": "Γ",
    "digamma": "ψ",
    "Ai": "Ai", "Bi": "Bi",
    "J_0": "J₀", "J_1": "J₁", "Y_0": "Y₀", "Y_1": "Y₁",
    "I_0": "I₀", "I_1": "I₁", "K_0": "K₀", "K_1": "K₁",
    "j_0": "j₀", "j_1": "j₁",
    "H_0": "H₀", "H_1": "H₁", "H_2": "H₂", "H_3": "H₃", "H_4": "H₄",
    "L_0": "L₀", "L_1": "L₁", "L_2": "L₂",
    "P_0": "P₀", "P_1": "P₁", "P_2": "P₂", "P_3": "P₃", "P_4": "P₄",
    "T_0": "T₀", "T_1": "T₁", "T_2": "T₂", "T_3": "T₃",
    "psi_0": "ψ₀", "psi_1": "ψ₁", "psi_2": "ψ₂",
    "R_10": "R₁₀", "R_20": "R₂₀", "R_21": "R₂₁",
    "heaviside": "Θ", "Theta": "Θ",
    "sinc": "sinc",
    "lorentzian": "L",
    "bose": "n_B", "fermi": "n_F", "planck": "B",
    "propagator": "G", "yukawa": "Y", "coulomb": "V",
}


def to_superscript(s: str) -> str:
    """Convert a string to Unicode superscript."""
    sup_map = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
        'n': 'ⁿ', 'i': 'ⁱ', '.': '·',
    }
    return ''.join(sup_map.get(c, c) for c in s)


def to_subscript(s: str) -> str:
    """Convert a string to Unicode subscript."""
    sub_map = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
        'a': 'ₐ', 'e': 'ₑ', 'h': 'ₕ', 'i': 'ᵢ', 'j': 'ⱼ',
        'k': 'ₖ', 'l': 'ₗ', 'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ',
        'p': 'ₚ', 'r': 'ᵣ', 's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ',
        'v': 'ᵥ', 'x': 'ₓ',
    }
    return ''.join(sub_map.get(c, c) for c in s)


def format_number_unicode(x: float, decimals: int = 2) -> str:
    """Format a number with Unicode minus sign and proper decimals."""
    if abs(x) < 10**(-decimals):
        return "0"
    if abs(x - round(x)) < 10**(-decimals - 1):
        return str(int(round(x))).replace("-", "−")
    return f"{x:.{decimals}f}".replace("-", "−")


@dataclass
class FormulaTerm:
    """A single term in a symbolic formula."""
    fn_name: str
    a: float  # input scale
    b: float  # input bias
    c: float  # output scale
    d: float  # output bias
    var_name: str = "x"

    def unicode(self, decimals: int = 2) -> str:
        """Render as Unicode string."""
        # Build inner expression: a*x + b
        inner = self._format_inner_unicode(decimals)

        # Apply function
        fn_unicode = UNICODE_FUNCTIONS.get(self.fn_name, self.fn_name)

        if self.fn_name == "x":
            f_inner = inner
        elif self.fn_name in ("x^2", "x^3", "x^4", "x^5"):
            power = self.fn_name[-1]
            f_inner = f"({inner}){to_superscript(power)}" if "+" in inner or "−" in inner else f"{inner}{to_superscript(power)}"
        elif self.fn_name == "x^0.5" or self.fn_name == "sqrt":
            f_inner = f"√({inner})" if "+" in inner or "−" in inner else f"√{inner}"
        elif self.fn_name.startswith("x^-"):
            power = self.fn_name[2:]
            f_inner = f"1/({inner}){to_superscript(power.replace('-', ''))}" if "+" in inner or "−" in inner else f"1/{inner}{to_superscript(power.replace('-', ''))}"
        elif self.fn_name == "gaussian":
            f_inner = f"e^(−({inner})²)" if "+" in inner or "−" in inner else f"e^(−{inner}²)"
        elif self.fn_name == "abs":
            f_inner = f"|{inner}|"
        elif self.fn_name == "sigmoid":
            f_inner = f"σ({inner})"
        else:
            f_inner = f"{fn_unicode}({inner})"

        # Outer: c * f(...) + d
        result = self._format_outer_unicode(f_inner, decimals)
        return result

    def _format_inner_unicode(self, decimals: int) -> str:
        """Format inner expression a*x + b."""
        if abs(self.a - 1.0) < 1e-6:
            inner = self.var_name
        elif abs(self.a + 1.0) < 1e-6:
            inner = f"−{self.var_name}"
        else:
            a_str = format_number_unicode(self.a, decimals)
            inner = f"{a_str}·{self.var_name}"

        if abs(self.b) > 10**(-decimals):
            b_str = format_number_unicode(abs(self.b), decimals)
            if self.b > 0:
                inner = f"({inner} + {b_str})"
            else:
                inner = f"({inner} − {b_str})"

        return inner

    def _format_outer_unicode(self, f_inner: str, decimals: int) -> str:
        """Format outer expression c * f(...) + d."""
        if abs(self.c - 1.0) < 1e-6:
            result = f_inner
        elif abs(self.c + 1.0) < 1e-6:
            result = f"−{f_inner}"
        else:
            c_str = format_number_unicode(self.c, decimals)
            result = f"{c_str}·{f_inner}"

        if abs(self.d) > 10**(-decimals):
            d_str = format_number_unicode(abs(self.d), decimals)
            if self.d > 0:
                result = f"{result} + {d_str}"
            else:
                result = f"{result} − {d_str}"

        return result

    def latex(self, decimals: int = 2) -> str:
        """Render as LaTeX string."""
        from .symbolic import SYMBOLIC_REGISTRY

        # Build inner expression
        inner = self._format_inner_latex(decimals)

        # Get LaTeX function representation
        if self.fn_name in SYMBOLIC_REGISTRY:
            latex_template = SYMBOLIC_REGISTRY[self.fn_name].latex
            f_inner = latex_template.replace("x", f"({inner})" if "+" in inner or "-" in inner else inner)
        else:
            f_inner = f"\\text{{{self.fn_name}}}({inner})"

        # Outer transformation
        result = self._format_outer_latex(f_inner, decimals)
        return result

    def _format_inner_latex(self, decimals: int) -> str:
        """Format inner expression for LaTeX."""
        if abs(self.a - 1.0) < 1e-6:
            inner = self.var_name
        elif abs(self.a + 1.0) < 1e-6:
            inner = f"-{self.var_name}"
        else:
            inner = f"{self.a:.{decimals}f} {self.var_name}"

        if abs(self.b) > 10**(-decimals):
            if self.b > 0:
                inner = f"{inner} + {self.b:.{decimals}f}"
            else:
                inner = f"{inner} - {abs(self.b):.{decimals}f}"

        return inner

    def _format_outer_latex(self, f_inner: str, decimals: int) -> str:
        """Format outer expression for LaTeX."""
        if abs(self.c - 1.0) < 1e-6:
            result = f_inner
        elif abs(self.c + 1.0) < 1e-6:
            result = f"-{f_inner}"
        else:
            result = f"{self.c:.{decimals}f} {f_inner}"

        if abs(self.d) > 10**(-decimals):
            if self.d > 0:
                result = f"{result} + {self.d:.{decimals}f}"
            else:
                result = f"{result} - {abs(self.d):.{decimals}f}"

        return result

    def typst(self, decimals: int = 2) -> str:
        """Render as Typst string."""
        from .symbolic import SYMBOLIC_REGISTRY

        # Build inner expression
        inner = self._format_inner_latex(decimals)  # Similar to LaTeX

        # Get Typst function representation
        if self.fn_name in SYMBOLIC_REGISTRY:
            typst_template = SYMBOLIC_REGISTRY[self.fn_name].typst
            f_inner = typst_template.replace("x", f"({inner})" if "+" in inner or "-" in inner else inner)
        else:
            f_inner = f'op("{self.fn_name}")({inner})'

        # Outer transformation
        result = self._format_outer_latex(f_inner, decimals)  # Similar formatting
        return result.replace("*", " dot ")

    def to_sympy(self):
        """Convert to SymPy expression."""
        try:
            import sympy
        except ImportError:
            raise ImportError("sympy required. Install with: pip install sympy")

        x = sympy.Symbol(self.var_name)
        inner = self.a * x + self.b

        # Map to sympy functions
        fn_map = {
            "x": lambda z: z,
            "x^2": lambda z: z**2,
            "x^3": lambda z: z**3,
            "x^4": lambda z: z**4,
            "x^0.5": sympy.sqrt,
            "sqrt": sympy.sqrt,
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tan": sympy.tan,
            "sinh": sympy.sinh,
            "cosh": sympy.cosh,
            "tanh": sympy.tanh,
            "exp": sympy.exp,
            "log": sympy.log,
            "abs": sympy.Abs,
            "sign": sympy.sign,
            "gaussian": lambda z: sympy.exp(-z**2),
            "sigmoid": lambda z: 1 / (1 + sympy.exp(-z)),
        }

        if self.fn_name in fn_map:
            f_expr = fn_map[self.fn_name](inner)
        else:
            # Create a generic function
            f = sympy.Function(self.fn_name)
            f_expr = f(inner)

        return self.c * f_expr + self.d


class FormulaRenderer:
    """Render symbolic formulas from a trained KAN model."""

    def __init__(self, model, var_names: Optional[List[str]] = None):
        """Initialize renderer with a model.

        Args:
            model: MultKAN model instance
            var_names: Optional variable names (default: x₀, x₁, ...)
        """
        self.model = model
        self.var_names = var_names or [f"x{to_subscript(str(i))}" for i in range(model.width[0])]
        self._terms = self._extract_terms()

    def _extract_terms(self) -> List[List[List[Optional[FormulaTerm]]]]:
        """Extract symbolic terms from model."""
        terms = []
        for l in range(self.model.depth):
            layer = self.model.layers[l]
            sym_layer = self.model.symbolic_funs[l]
            layer_terms = []

            for j in range(layer.out_dim):
                j_terms = []
                for i in range(layer.in_dim):
                    if sym_layer.is_symbolic(i, j):
                        fn_name = sym_layer.fns_name[i][j]
                        a = float(sym_layer.affine_a[i, j])
                        b = float(sym_layer.affine_b[i, j])
                        c = float(sym_layer.affine_c[i, j])
                        d = float(sym_layer.affine_d[i, j])
                        j_terms.append(FormulaTerm(fn_name, a, b, c, d))
                    else:
                        j_terms.append(None)  # Spline, not symbolic
                layer_terms.append(j_terms)
            terms.append(layer_terms)

        return terms

    def unicode(self, decimals: int = 2, show_layers: bool = False) -> str:
        """Render formula as Unicode string for terminal display.

        Args:
            decimals: Number of decimal places
            show_layers: If True, show intermediate layer outputs
        """
        expressions = [self.var_names.copy()]

        lines = []
        if show_layers:
            lines.append("━" * 50)
            lines.append("  Input: " + ", ".join(self.var_names))

        for l, layer_terms in enumerate(self._terms):
            layer_expressions = []

            for j, j_terms in enumerate(layer_terms):
                terms_strs = []
                for i, term in enumerate(j_terms):
                    if term is not None:
                        term.var_name = expressions[l][i]
                        terms_strs.append(term.unicode(decimals))
                    else:
                        terms_strs.append(f"φ{to_subscript(f'{l},{i},{j}')}({expressions[l][i]})")

                if len(terms_strs) == 1:
                    expr = terms_strs[0]
                else:
                    expr = " + ".join(terms_strs)
                    if len(terms_strs) > 1:
                        expr = f"({expr})"

                layer_expressions.append(expr)

            expressions.append(layer_expressions)

            if show_layers:
                lines.append(f"\n  Layer {l}:")
                for j, expr in enumerate(layer_expressions):
                    lines.append(f"    y{to_subscript(str(j))} = {expr}")

        # Final output
        final = expressions[-1]
        result = final[0] if len(final) == 1 else ", ".join(final)

        if show_layers:
            lines.append("\n" + "━" * 50)
            lines.append(f"  f({', '.join(self.var_names)}) = {result}")
            lines.append("━" * 50)
            return "\n".join(lines)

        return f"f({', '.join(self.var_names)}) = {result}"

    def latex(self, decimals: int = 2, equation_env: bool = False) -> str:
        """Render formula as LaTeX string.

        Args:
            decimals: Number of decimal places
            equation_env: If True, wrap in equation environment
        """
        expressions = [[f"x_{{{i}}}" for i in range(len(self.var_names))]]

        for l, layer_terms in enumerate(self._terms):
            layer_expressions = []

            for j, j_terms in enumerate(layer_terms):
                terms_strs = []
                for i, term in enumerate(j_terms):
                    if term is not None:
                        term.var_name = expressions[l][i]
                        terms_strs.append(term.latex(decimals))
                    else:
                        terms_strs.append(f"\\phi_{{{l},{i},{j}}}({expressions[l][i]})")

                expr = terms_strs[0] if len(terms_strs) == 1 else " + ".join(terms_strs)
                layer_expressions.append(expr)

            expressions.append(layer_expressions)

        final = expressions[-1]
        var_str = ", ".join(f"x_{{{i}}}" for i in range(len(self.var_names)))
        result = final[0] if len(final) == 1 else ", ".join(final)

        formula = f"f({var_str}) = {result}"

        if equation_env:
            return f"\\begin{{equation}}\n{formula}\n\\end{{equation}}"
        return formula

    def typst(self, decimals: int = 2, math_block: bool = False) -> str:
        """Render formula as Typst string.

        Args:
            decimals: Number of decimal places
            math_block: If True, wrap in math block
        """
        expressions = [[f"x_({i})" for i in range(len(self.var_names))]]

        for l, layer_terms in enumerate(self._terms):
            layer_expressions = []

            for j, j_terms in enumerate(layer_terms):
                terms_strs = []
                for i, term in enumerate(j_terms):
                    if term is not None:
                        term.var_name = expressions[l][i]
                        terms_strs.append(term.typst(decimals))
                    else:
                        terms_strs.append(f"phi_({l},{i},{j})({expressions[l][i]})")

                expr = terms_strs[0] if len(terms_strs) == 1 else " + ".join(terms_strs)
                layer_expressions.append(expr)

            expressions.append(layer_expressions)

        final = expressions[-1]
        var_str = ", ".join(f"x_({i})" for i in range(len(self.var_names)))
        result = final[0] if len(final) == 1 else ", ".join(final)

        formula = f"f({var_str}) = {result}"

        if math_block:
            return f"$ {formula} $"
        return formula

    def to_sympy(self, simplify: bool = True):
        """Convert to SymPy expression.

        Args:
            simplify: If True, simplify the expression

        Returns:
            sympy.Expr or list of sympy.Expr
        """
        try:
            import sympy
        except ImportError:
            raise ImportError("sympy required. Install with: pip install sympy")

        symbols = [sympy.Symbol(f"x{i}") for i in range(len(self.var_names))]
        expressions = [symbols.copy()]

        for l, layer_terms in enumerate(self._terms):
            layer_expressions = []

            for j, j_terms in enumerate(layer_terms):
                expr = sympy.Integer(0)
                for i, term in enumerate(j_terms):
                    if term is not None:
                        term.var_name = f"__TEMP_{l}_{i}__"
                        term_expr = term.to_sympy()
                        # Substitute the actual expression
                        temp_sym = sympy.Symbol(f"__TEMP_{l}_{i}__")
                        term_expr = term_expr.subs(temp_sym, expressions[l][i])
                        expr = expr + term_expr
                    else:
                        # Non-symbolic edge: use a function placeholder
                        phi = sympy.Function(f"phi_{l}_{i}_{j}")
                        expr = expr + phi(expressions[l][i])

                layer_expressions.append(expr)

            expressions.append(layer_expressions)

        final = expressions[-1]
        if simplify:
            final = [sympy.simplify(e) for e in final]

        return final[0] if len(final) == 1 else final

    def display(self, format: str = "auto"):
        """Display formula in appropriate format for the environment.

        Args:
            format: "auto", "unicode", "latex", "html"
        """
        if format == "auto":
            # Detect environment
            try:
                from IPython.display import display, Math, HTML
                # We're in Jupyter/IPython
                latex_str = self.latex()
                display(Math(latex_str))
                return
            except ImportError:
                pass

            # Terminal output
            print(self.unicode(show_layers=True))
            return

        if format == "unicode":
            print(self.unicode(show_layers=True))
        elif format == "latex":
            print(self.latex(equation_env=True))
        elif format == "html":
            # For web contexts
            latex_str = self.latex()
            print(f'<span class="math">{latex_str}</span>')

    def summary(self) -> str:
        """Print a summary of the symbolic formula with stats."""
        lines = []
        lines.append("\n╔══════════════════════════════════════════════════╗")
        lines.append("║           SYMBOLIC FORMULA SUMMARY                ║")
        lines.append("╚══════════════════════════════════════════════════╝\n")

        # Count symbolic vs spline edges
        n_symbolic = 0
        n_spline = 0
        fn_counts: Dict[str, int] = {}

        for layer_terms in self._terms:
            for j_terms in layer_terms:
                for term in j_terms:
                    if term is not None:
                        n_symbolic += 1
                        fn_counts[term.fn_name] = fn_counts.get(term.fn_name, 0) + 1
                    else:
                        n_spline += 1

        lines.append(f"  Symbolic edges: {n_symbolic}")
        lines.append(f"  Spline edges:   {n_spline}")
        lines.append(f"  Symbolic ratio: {n_symbolic / (n_symbolic + n_spline) * 100:.1f}%\n")

        if fn_counts:
            lines.append("  Functions used:")
            for fn, count in sorted(fn_counts.items(), key=lambda x: -x[1]):
                fn_unicode = UNICODE_FUNCTIONS.get(fn, fn)
                lines.append(f"    {fn_unicode}: {count}")

        lines.append("\n" + "─" * 52)
        lines.append("\n  Unicode formula:")
        lines.append("  " + self.unicode())

        lines.append("\n  LaTeX formula:")
        lines.append("  " + self.latex())

        lines.append("\n" + "─" * 52 + "\n")

        return "\n".join(lines)


def render_formula(
    model,
    format: str = "unicode",
    var_names: Optional[List[str]] = None,
    decimals: int = 2,
    **kwargs,
) -> str:
    """Convenience function to render a model's formula.

    Args:
        model: MultKAN model instance
        format: "unicode", "latex", "typst", or "sympy"
        var_names: Optional variable names
        decimals: Decimal places for coefficients
        **kwargs: Additional arguments for the format method

    Returns:
        Formatted formula string (or sympy.Expr if format="sympy")
    """
    renderer = FormulaRenderer(model, var_names)

    if format == "unicode":
        return renderer.unicode(decimals, **kwargs)
    elif format == "latex":
        return renderer.latex(decimals, **kwargs)
    elif format == "typst":
        return renderer.typst(decimals, **kwargs)
    elif format == "sympy":
        return renderer.to_sympy(**kwargs)
    elif format == "summary":
        return renderer.summary()
    else:
        raise ValueError(f"Unknown format: {format}. Use 'unicode', 'latex', 'typst', 'sympy', or 'summary'")


def print_formula_box(formula: str, title: str = "Formula") -> None:
    """Print a formula in a nice Unicode box."""
    width = max(len(formula) + 4, len(title) + 4, 40)

    print("┌" + "─" * width + "┐")
    print("│" + title.center(width) + "│")
    print("├" + "─" * width + "┤")
    print("│  " + formula.ljust(width - 2) + "│")
    print("└" + "─" * width + "┘")
