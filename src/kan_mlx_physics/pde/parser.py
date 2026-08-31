"""Multi-format PDE Parser with SymPy backend.

Parses PDE equations from multiple notation formats:
- Unicode: ∇²ψ, ∂ψ/∂x, □φ
- LaTeX: \\nabla^2 \\psi, \\frac{\\partial \\psi}{\\partial x}
- Subscript: u_xx, psi_t
- SymPy: Derivative(u, x, 2)
- Natural: laplacian(u), d2u/dx2

All formats are normalized to a canonical string that SymPy can parse,
then converted to the PDEExpr intermediate representation.

Example:
    >>> parser = PDEParser()
    >>> parsed = parser.parse("-∇²ψ/2 + V(x)ψ = Eψ")
    >>> print(parsed.lhs)  # PDEExpr tree
    >>> print(parsed.rhs)  # PDEExpr tree
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass

from .expr import (
    PDEExpr, ParsedPDE, Variable, Coordinate, Parameter, Constant,
    Derivative, Laplacian, Gradient, Dalembert, StarProduct, Hamiltonian,
    BinaryOp, UnaryOp, FunctionCall, collect_nodes, walk_expr
)

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    sp = None


# =============================================================================
# FORMAT DETECTION
# =============================================================================

class FormatDetector:
    """Detect the notation format of an equation string."""

    # Patterns for each format
    PATTERNS = {
        'unicode': [
            r'[∇∂□⋆∆]',           # Unicode operators
            r'[ψφΨΦχ]',            # Greek letters
            r'[ℏ]',                # hbar symbol
        ],
        'latex': [
            r'\\nabla',
            r'\\partial',
            r'\\frac\{',
            r'\\psi|\\phi|\\Psi',
            r'\\hbar',
            r'\^{?\d+}?',          # Superscripts like ^2 or ^{2}
        ],
        'typst': [
            r'\$[^$]+\$',              # Typst inline math $...$
            r'frac\([^)]+,[^)]+\)',    # Typst frac(a, b)
            r'diff\s*$',               # Typst diff operator
            r'laplacian\(',            # Typst laplacian()
            r'partial_[xyzt]',         # Typst partial subscript
            r'nabla\^2(?!\s*\\)',      # Typst nabla^2 (not LaTeX \nabla)
            r'arrow\.r',               # Typst arrow syntax
        ],
        'subscript': [
            r'[a-zA-Z]_[xyztapr]{1,3}',  # u_xx, psi_t, etc.
            r"[a-zA-Z]''+",              # u'', psi'
        ],
        'sympy': [
            r'Derivative\s*\(',
            r'Function\s*\(',
            r'Symbol\s*\(',
            r'sp\.',
        ],
    }

    def detect(self, equation: str) -> str:
        """Detect the format of an equation string.

        Returns:
            Format name: 'unicode', 'latex', 'subscript', 'sympy', or 'natural'.
        """
        scores = {fmt: 0 for fmt in self.PATTERNS}

        for fmt, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, equation):
                    scores[fmt] += 1

        # Return format with highest score, default to 'natural'
        max_score = max(scores.values())
        if max_score == 0:
            return 'natural'

        for fmt, score in scores.items():
            if score == max_score:
                return fmt

        return 'natural'


# =============================================================================
# NORMALIZERS
# =============================================================================

class UnicodeNormalizer:
    """Normalize Unicode notation to SymPy-compatible strings."""

    # Unicode to ASCII mappings
    OPERATORS = {
        '∇²': 'laplacian',
        '∇^2': 'laplacian',
        '∆': 'laplacian',
        '∇': 'grad',
        '□': 'dalembert',
        '⋆': 'star',
        '★': 'star',
        '∂': 'd',
    }

    GREEK = {
        'ψ': 'psi', 'φ': 'phi', 'Ψ': 'Psi', 'Φ': 'Phi',
        'χ': 'chi', 'ω': 'omega', 'θ': 'theta',
        'α': 'alpha', 'β': 'beta', 'γ': 'gamma',
        'λ': 'lambda_', 'Λ': 'Lambda',
        'ℏ': 'hbar', 'π': 'pi',
        'τ': 'tau', 'η': 'eta', 'ξ': 'xi',
    }

    SUPERSCRIPTS = {
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    }

    SUBSCRIPTS = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        'ₓ': 'x', 'ₜ': 't', 'ᵣ': 'r', 'ₐ': 'a',
    }

    def normalize(self, eq: str) -> str:
        """Normalize Unicode equation to SymPy-compatible string."""
        result = eq

        # Replace operators
        for unicode_op, ascii_op in self.OPERATORS.items():
            result = result.replace(unicode_op, ascii_op)

        # Replace Greek letters
        for greek, ascii_name in self.GREEK.items():
            result = result.replace(greek, ascii_name)

        # Replace superscripts
        for sup, digit in self.SUPERSCRIPTS.items():
            result = result.replace(sup, f'^{digit}')

        # Replace subscripts
        for sub, char in self.SUBSCRIPTS.items():
            result = result.replace(sub, f'_{char}')

        # Handle partial derivative notation: ∂u/∂x → Derivative(u, x)
        result = self._convert_partial_derivatives(result)

        # Handle Laplacian: laplacian(u) stays as is (handled in parsing)

        return result

    def _convert_partial_derivatives(self, eq: str) -> str:
        """Convert ∂u/∂x notation to Derivative(u, x)."""
        # Pattern: d^n u / d x^n or d u / d x
        # Also handles: du/dx, d²u/dx²

        # Second order: d²u/dx² or d^2 u/dx^2
        pattern_2nd = r'd\^?2?\s*(\w+)\s*/\s*d\s*(\w+)\^?2?'
        result = re.sub(pattern_2nd, r'Derivative(\1, \2, 2)', eq)

        # First order: du/dx
        pattern_1st = r'd\s*(\w+)\s*/\s*d\s*(\w+)(?!\^)'
        result = re.sub(pattern_1st, r'Derivative(\1, \2)', result)

        return result


class LatexNormalizer:
    """Normalize LaTeX notation to SymPy-compatible strings.

    Supports standard LaTeX math notation:
    - Fractions: \\frac{a}{b}
    - Partial derivatives: \\frac{\\partial u}{\\partial x}, \\partial_x u
    - Total derivatives: \\frac{d f}{d t}
    - Greek letters: \\psi, \\phi, \\alpha, etc.
    - Operators: \\nabla^2, \\Box, \\star
    - physics package: \\pdv{f}{x}, \\dv{f}{t}
    """

    def normalize(self, eq: str) -> str:
        """Normalize LaTeX equation to SymPy-compatible string."""
        result = eq

        # Remove LaTeX commands
        result = result.replace('\\,', ' ')
        result = result.replace('\\;', ' ')
        result = result.replace('\\quad', ' ')
        result = result.replace('\\qquad', ' ')
        result = result.replace('\\left', '')
        result = result.replace('\\right', '')
        result = result.replace('\\cdot', '*')
        result = result.replace('\\times', '*')

        # Handle physics package derivatives FIRST (before other conversions)
        result = self._convert_physics_package(result)

        # Convert \frac{\partial u}{\partial x} to Derivative(u, x)
        result = self._convert_frac_partial_derivatives(result)

        # Convert \frac{d f}{d t} to Derivative(f, t) - total derivatives
        result = self._convert_frac_total_derivatives(result)

        # Convert subscripted partials: \partial_x u, \partial_x^2 u
        result = self._convert_subscript_partials(result)

        # Operators - convert to canonical form with parentheses
        # \nabla^2 \psi → laplacian(psi)
        result = re.sub(r'\\nabla\^{?2}?\s*\\?(\w+)', r'laplacian(\1)', result)
        result = result.replace('\\nabla^2', 'laplacian')
        result = result.replace('\\nabla^{2}', 'laplacian')

        # \Delta \psi → laplacian(psi)
        result = re.sub(r'\\Delta\s+\\?(\w+)', r'laplacian(\1)', result)
        result = result.replace('\\Delta', 'laplacian')

        # \nabla (gradient) - keep as grad for now
        result = result.replace('\\nabla', 'grad')

        # \Box \phi → dalembert(phi), \square \phi → dalembert(phi)
        result = re.sub(r'\\Box\s+\\?(\w+)', r'dalembert(\1)', result)
        result = re.sub(r'\\square\s+\\?(\w+)', r'dalembert(\1)', result)
        result = result.replace('\\Box', 'dalembert')
        result = result.replace('\\square', 'dalembert')

        result = result.replace('\\star', 'star')

        # Greek letters (comprehensive list)
        greek_map = {
            '\\psi': 'psi', '\\Psi': 'Psi', '\\phi': 'phi', '\\Phi': 'Phi',
            '\\chi': 'chi', '\\omega': 'omega', '\\Omega': 'Omega',
            '\\theta': 'theta', '\\Theta': 'Theta',
            '\\hbar': 'hbar', '\\pi': 'pi',
            '\\lambda': 'lambda_', '\\Lambda': 'Lambda',
            '\\alpha': 'alpha', '\\beta': 'beta', '\\gamma': 'gamma', '\\Gamma': 'Gamma',
            '\\delta': 'delta', '\\Delta': 'Delta',
            '\\epsilon': 'epsilon', '\\varepsilon': 'epsilon',
            '\\zeta': 'zeta', '\\eta': 'eta',
            '\\kappa': 'kappa', '\\mu': 'mu', '\\nu': 'nu',
            '\\xi': 'xi', '\\Xi': 'Xi',
            '\\rho': 'rho', '\\sigma': 'sigma', '\\Sigma': 'Sigma',
            '\\tau': 'tau', '\\upsilon': 'upsilon',
            '\\varphi': 'phi', '\\partial': 'partial',
        }
        for latex_sym, ascii_sym in greek_map.items():
            result = result.replace(latex_sym, ascii_sym)

        # Convert \frac{a}{b} to (a)/(b) - general fractions
        result = self._convert_fractions(result)

        # Remove remaining braces
        result = result.replace('{', '(').replace('}', ')')

        # Clean up superscripts
        result = re.sub(r'\^\((\d+)\)', r'**\1', result)
        result = re.sub(r'\^(\d)', r'**\1', result)

        # Handle implicit multiplication
        result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)
        result = re.sub(r'\)\s*\(', r')*(', result)
        result = re.sub(r'\)\s*([a-zA-Z])', r')*\1', result)

        return result

    def _convert_physics_package(self, eq: str) -> str:
        """Convert LaTeX physics package notation.

        Handles:
        - \\pdv{f}{x} → Derivative(f, x)
        - \\pdv[2]{f}{x} → Derivative(f, x, 2)
        - \\pdv{f}{x}{y} → mixed partial
        - \\dv{f}{t} → Derivative(f, t)
        - \\dv[2]{f}{t} → Derivative(f, t, 2)
        - \\dd{x} → dx (differential)
        - \\dd[n]{x} → d^n x
        """
        result = eq

        # \pdv[n]{f}{x} - partial derivative with order
        pattern_pdv_order = r'\\pdv\[(\d+)\]\{(\w+)\}\{(\w+)\}'
        result = re.sub(pattern_pdv_order, r'Derivative(\2, \3, \1)', result)

        # \pdv{f}{x} - first order partial
        pattern_pdv = r'\\pdv\{(\w+)\}\{(\w+)\}'
        result = re.sub(pattern_pdv, r'Derivative(\1, \2)', result)

        # \dv[n]{f}{t} - total derivative with order
        pattern_dv_order = r'\\dv\[(\d+)\]\{(\w+)\}\{(\w+)\}'
        result = re.sub(pattern_dv_order, r'Derivative(\2, \3, \1)', result)

        # \dv{f}{t} - first order total derivative
        pattern_dv = r'\\dv\{(\w+)\}\{(\w+)\}'
        result = re.sub(pattern_dv, r'Derivative(\1, \2)', result)

        # \dd{x} - differential (just strip it, it's for display)
        # \dd[n]{x} → d^n x
        result = re.sub(r'\\dd\[(\d+)\]\{(\w+)\}', r'd^\1\2', result)
        result = re.sub(r'\\dd\{(\w+)\}', r'd\1', result)
        result = re.sub(r'\\dd\s+(\w+)', r'd\1', result)  # \dd x without braces

        return result

    def _convert_frac_partial_derivatives(self, eq: str) -> str:
        """Convert LaTeX fraction partial derivatives to Derivative()."""
        # \frac{\partial^2 u}{\partial x^2}
        pattern = r'\\frac\{\\partial\^?{?(\d*)}?\s*(\w+)\}\{\\partial\s*(\w+)\^?{?(\d*)}?\}'

        def replace_deriv(m):
            order1 = m.group(1) or '1'
            var = m.group(2)
            coord = m.group(3)
            order2 = m.group(4) or '1'
            order = max(int(order1), int(order2))
            if order == 1:
                return f'Derivative({var}, {coord})'
            return f'Derivative({var}, {coord}, {order})'

        return re.sub(pattern, replace_deriv, eq)

    def _convert_frac_total_derivatives(self, eq: str) -> str:
        """Convert LaTeX fraction total derivatives to Derivative().

        Handles: \\frac{d f}{d t}, \\frac{d^2 f}{d t^2}
        """
        # \frac{d^n f}{d t^n}
        pattern = r'\\frac\{d\^?{?(\d*)}?\s*(\w+)\}\{d\s*(\w+)\^?{?(\d*)}?\}'

        def replace_deriv(m):
            order1 = m.group(1) or '1'
            var = m.group(2)
            coord = m.group(3)
            order2 = m.group(4) or '1'
            order = max(int(order1), int(order2))
            if order == 1:
                return f'Derivative({var}, {coord})'
            return f'Derivative({var}, {coord}, {order})'

        return re.sub(pattern, replace_deriv, eq)

    def _convert_subscript_partials(self, eq: str) -> str:
        """Convert subscripted partial notation.

        Handles:
        - \\partial_x u → Derivative(u, x)
        - \\partial_x^2 u → Derivative(u, x, 2)
        - \\partial^2_x u → Derivative(u, x, 2)
        """
        result = eq
        coord_pat = r'[xyztatprq]'

        # \partial_x^n u
        pattern1 = rf'\\partial_({coord_pat})\^{{?(\d+)}}?\s*(\w+)'
        result = re.sub(pattern1, r'Derivative(\3, \1, \2)', result)

        # \partial^n_x u
        pattern2 = rf'\\partial\^{{?(\d+)}}?_({coord_pat})\s*(\w+)'
        result = re.sub(pattern2, r'Derivative(\3, \2, \1)', result)

        # \partial_x u (first order)
        pattern3 = rf'\\partial_({coord_pat})\s+(\w+)'
        result = re.sub(pattern3, r'Derivative(\2, \1)', result)

        return result

    def _convert_fractions(self, eq: str) -> str:
        """Convert LaTeX fractions to division."""
        # Simple \frac{a}{b} → (a)/(b)
        pattern = r'\\frac\{([^{}]+)\}\{([^{}]+)\}'
        while re.search(pattern, eq):
            eq = re.sub(pattern, r'(\1)/(\2)', eq)
        return eq


class SubscriptNormalizer:
    """Normalize subscript notation (u_xx, psi_t) to SymPy."""

    def normalize(self, eq: str) -> str:
        """Normalize subscript notation to Derivative()."""
        result = eq

        # Pattern: u_xx, psi_ttt, phi_xy
        # First handle double/triple subscripts (second+ derivatives)
        pattern = r'(\w+)_([xyztapr])(\2+)'

        def replace_repeated(m):
            var = m.group(1)
            coord = m.group(2)
            count = 1 + len(m.group(3))
            return f'Derivative({var}, {coord}, {count})'

        result = re.sub(pattern, replace_repeated, result)

        # Single subscript (first derivative)
        pattern_single = r'(\w+)_([xyztapr])(?![xyztapr])'
        result = re.sub(pattern_single, r'Derivative(\1, \2)', result)

        # Prime notation: u'', psi'
        result = re.sub(r"(\w+)''+", lambda m: f'Derivative({m.group(1)}, x, {len(m.group(0)) - len(m.group(1))})', result)
        result = re.sub(r"(\w+)'(?!')", r'Derivative(\1, x)', result)

        return result


class NaturalNormalizer:
    """Normalize natural/plain notation."""

    def normalize(self, eq: str) -> str:
        """Normalize natural notation."""
        result = eq

        # d2u/dx2 → Derivative(u, x, 2)
        result = re.sub(r'd(\d*)(\w+)/d(\w+)(\d*)',
                       lambda m: f'Derivative({m.group(2)}, {m.group(3)}, {max(int(m.group(1) or 1), int(m.group(4) or 1))})',
                       result)

        # nabla^2 → laplacian
        result = result.replace('nabla^2', 'laplacian')
        result = result.replace('nabla2', 'laplacian')

        # Handle implicit multiplication
        # Add * between number and variable: 2u → 2*u
        result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)

        # Add * between variable and (: u(x) stays, but x u → x*u
        result = re.sub(r'([a-zA-Z])(\s+)([a-zA-Z])', r'\1*\3', result)

        return result


class TypstNormalizer:
    """Normalize Typst math notation to SymPy-compatible strings.

    Typst math notation reference (from official docs):
    - Math delimiters: $ ... $ (inline) or $$ ... $$ (display)
    - Fractions: frac(a, b) or a / b (Typst auto-renders as fraction)
    - Derivatives: 'diff' (∂) is the partial symbol, 'dif' for differential d
    - Greek letters: psi, phi, alpha, etc. (no backslash needed)
    - Operators: nabla^2, square (d'Alembertian)
    - Superscripts: x^2, x^(n+1)
    - Subscripts: x_i, x_(i+1)
    - Special symbols: planck.reduce (ℏ), dot.c (·), arrow.r (→)

    Supports common package syntaxes:
    - physica: pdv(f, x) → ∂f/∂x, pdv(f, x, 2) → ∂²f/∂x²
    - diverential: dv(f, x) → df/dx

    Example Typst equations:
        $ -nabla^2 psi / 2 + V(x) psi = E psi $
        $ frac(diff^2 u, diff x^2) = 0 $
        $ diff u / diff x + a u = 0 $
        $ pdv(psi, x, 2) + k^2 psi = 0 $
    """

    # Typst Greek letters (no backslash needed in Typst)
    GREEK = {
        'psi': 'psi', 'phi': 'phi', 'Psi': 'Psi', 'Phi': 'Phi',
        'chi': 'chi', 'omega': 'omega', 'theta': 'theta',
        'alpha': 'alpha', 'beta': 'beta', 'gamma': 'gamma',
        'lambda': 'lambda_', 'Lambda': 'Lambda',
        'hbar': 'hbar', 'planck.reduce': 'hbar',
        'pi': 'pi', 'tau': 'tau', 'eta': 'eta', 'xi': 'xi',
        'epsilon': 'epsilon', 'delta': 'delta', 'Delta': 'Delta',
        'sigma': 'sigma', 'Sigma': 'Sigma',
        'mu': 'mu', 'nu': 'nu', 'rho': 'rho',
    }

    def normalize(self, eq: str) -> str:
        """Normalize Typst equation to SymPy-compatible string."""
        result = eq

        # Remove Typst math delimiters
        result = re.sub(r'^\s*\$+\s*', '', result)
        result = re.sub(r'\s*\$+\s*$', '', result)

        # Remove Typst spacing commands (word boundaries to avoid false matches)
        result = re.sub(r'\bthin\b', ' ', result)
        result = re.sub(r'\bmed\b', ' ', result)
        result = re.sub(r'\bthick\b', ' ', result)
        result = re.sub(r'\bquad\b', ' ', result)
        result = re.sub(r'\bwide\b', ' ', result)

        # Handle Typst special symbols
        result = result.replace('planck.reduce', 'hbar')
        result = result.replace('dot.c', '*')  # centered dot → multiplication
        result = result.replace('times', '*')

        # Handle Typst's frac(a, b) notation FIRST (before diff conversion)
        result = self._convert_typst_fractions(result)

        # Handle physica package: pdv(f, x) or pdv(f, x, 2)
        result = self._convert_physica_derivatives(result)

        # Handle diverential package: dv(f, x) or dv(f, x, 2)
        result = self._convert_diverential_derivatives(result)

        # Handle Typst's diff/dif notation for derivatives
        # In Typst: 'diff' is ∂ (partial), 'dif' is d (differential)
        # diff^2 u / diff x^2 → Derivative(u, x, 2)
        result = self._convert_diff_derivatives(result)

        # Handle partial_x notation (subscript style)
        result = self._convert_partial_subscripts(result)

        # Handle Typst operators - nabla^2 followed by variable
        # nabla^2 psi → laplacian(psi)
        result = re.sub(r'nabla\^2\s+(\w+)', r'laplacian(\1)', result)
        result = result.replace('nabla^2', 'laplacian')  # Fallback

        # Handle standalone laplacian followed by variable
        # laplacian psi → laplacian(psi) (if not already parenthesized)
        result = re.sub(r'laplacian\s+(\w+)(?!\s*\()', r'laplacian(\1)', result)

        # square phi → dalembert(phi)
        result = re.sub(r'square\s+(\w+)', r'dalembert(\1)', result)
        result = result.replace('square', 'dalembert')  # Fallback

        result = result.replace('star', 'star')

        # Handle Typst's sqrt
        result = re.sub(r'sqrt\(([^)]+)\)', r'sqrt(\1)', result)

        # Handle Typst's root(n, x) → x^(1/n)
        result = re.sub(r'root\((\d+),\s*([^)]+)\)', r'(\2)**(1/\1)', result)

        # Clean up parentheses in subscripts: x_(i) → x_i for single char
        result = re.sub(r'_\((\w)\)', r'_\1', result)

        # Clean up parentheses in superscripts for single digits: x^(2) → x^2
        result = re.sub(r'\^\((\d)\)', r'^\1', result)

        # Handle implicit multiplication (Typst allows spaces between atoms)
        # IMPORTANT: Apply after derivative conversion to avoid breaking patterns

        # Add * between closing paren and opening paren or letter
        result = re.sub(r'\)\s*\(', r')*(', result)
        result = re.sub(r'\)\s*([a-zA-Z])', r')*\1', result)

        # Add * between number and variable: 2 psi → 2*psi, 2psi → 2*psi
        result = re.sub(r'(\d)\s+([a-zA-Z])', r'\1*\2', result)
        result = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', result)

        # Handle space-multiplication between identifiers: V(x) psi → V(x)*psi
        # But avoid breaking function calls
        result = re.sub(r'(\w)\s+([a-zA-Z])(?!\s*\()', r'\1*\2', result)

        # Convert ^ to ** for SymPy
        result = re.sub(r'\^(\d+)', r'**\1', result)
        result = re.sub(r'\^\(([^)]+)\)', r'**(\1)', result)

        return result

    def _convert_physica_derivatives(self, eq: str) -> str:
        """Convert physica package pdv() notation to Derivative().

        Handles:
        - pdv(f, x) → Derivative(f, x)
        - pdv(f, x, 2) → Derivative(f, x, 2)
        - pdv(f, x, y) → mixed partial (simplified to sequential)
        """
        result = eq

        # pdv(f, x, n) where n is a number
        pattern_order = r'pdv\((\w+),\s*(\w+),\s*(\d+)\)'
        result = re.sub(pattern_order, r'Derivative(\1, \2, \3)', result)

        # pdv(f, x) - first order
        pattern_first = r'pdv\((\w+),\s*(\w+)\)'
        result = re.sub(pattern_first, r'Derivative(\1, \2)', result)

        return result

    def _convert_diverential_derivatives(self, eq: str) -> str:
        """Convert diverential package dv() notation to Derivative().

        Handles:
        - dv(f, x) → Derivative(f, x)
        - dv(f, x, 2) → Derivative(f, x, 2)
        """
        result = eq

        # dv(f, x, n) where n is a number
        pattern_order = r'dv\((\w+),\s*(\w+),\s*(\d+)\)'
        result = re.sub(pattern_order, r'Derivative(\1, \2, \3)', result)

        # dv(f, x) - first order
        pattern_first = r'dv\((\w+),\s*(\w+)\)'
        result = re.sub(pattern_first, r'Derivative(\1, \2)', result)

        return result

    def _convert_diff_derivatives(self, eq: str) -> str:
        """Convert Typst diff/dif/partial notation to Derivative().

        In Typst math:
        - 'diff' renders as ∂ (partial derivative symbol)
        - 'dif' renders as d (differential, upright)
        - 'partial' also works as synonym for diff

        Handles:
        - diff^2 u / diff x^2 → Derivative(u, x, 2)
        - diff u / diff x → Derivative(u, x)
        - dif u / dif x → Derivative(u, x)  (ordinary derivative)
        - (diff^2 psi) / (diff x^2) → Derivative(psi, x, 2)
        - After frac conversion: ((diff^2 u)/(diff x^2)) → Derivative(u, x, 2)
        """
        result = eq

        # All derivative symbols to match: diff, dif, partial
        deriv_sym = r'(?:diff|dif|partial)'

        # Pattern 1: Standard fraction-style derivatives (after frac conversion)
        # ((diff^2 u)/(diff x^2)) or ((partial^2 u)/(partial x^2))
        pattern_frac = rf'\(\({deriv_sym}\^?(\d*)\s*(\w+)\)/\({deriv_sym}\s*(\w+)\^?(\d*)\)\)'

        def replace_deriv(m):
            order1 = m.group(1) or '1'
            var = m.group(2)
            coord = m.group(3)
            order2 = m.group(4) or '1'
            order = max(int(order1), int(order2))
            if order == 1:
                return f'Derivative({var}, {coord})'
            return f'Derivative({var}, {coord}, {order})'

        result = re.sub(pattern_frac, replace_deriv, result)

        # Pattern 2: Plain fraction style: diff^n var / diff coord^n
        # With optional parentheses around parts
        pattern_plain = rf'\(?\s*{deriv_sym}\^?(\d*)\s*(\w+)\s*\)?\s*/\s*\(?\s*{deriv_sym}\s*(\w+)\^?(\d*)\s*\)?'
        result = re.sub(pattern_plain, replace_deriv, result)

        return result

    def _convert_typst_fractions(self, eq: str) -> str:
        """Convert Typst frac(a, b) to (a)/(b).

        Also handles nested fractions.
        """
        result = eq

        # Handle frac(a, b) → (a)/(b)
        # Need to handle nested parentheses properly
        max_iterations = 10
        for _ in range(max_iterations):
            # Match frac( with balanced content
            match = re.search(r'frac\(', result)
            if not match:
                break

            start = match.start()
            paren_start = match.end() - 1  # Position of opening (

            # Find matching closing paren and the comma separator
            depth = 1
            comma_pos = None
            i = paren_start + 1
            while i < len(result) and depth > 0:
                if result[i] == '(':
                    depth += 1
                elif result[i] == ')':
                    depth -= 1
                elif result[i] == ',' and depth == 1:
                    comma_pos = i
                i += 1

            if comma_pos is not None and depth == 0:
                end = i
                numerator = result[paren_start + 1:comma_pos].strip()
                denominator = result[comma_pos + 1:end - 1].strip()
                result = result[:start] + f'(({numerator})/({denominator}))' + result[end:]
            else:
                break

        return result

    def _convert_partial_subscripts(self, eq: str) -> str:
        """Convert Typst partial_x notation to derivatives.

        Handles:
        - partial_x^2 u → Derivative(u, x, 2)
        - partial_x u → Derivative(u, x)
        - partial^2_x u → Derivative(u, x, 2)
        - accent(partial,<-)_x → left-acting partial (treat as partial_x)
        """
        result = eq

        # Coordinate pattern: x, y, z, t, a, p, r, q (phase space coords)
        coord_pat = r'[xyztatprq]'

        # Handle Typst accent notation: accent(partial,<-)_x → partial_x
        # This is used for left-acting operators in Typst
        result = re.sub(rf'accent\s*\(\s*partial\s*,\s*[<>]\s*-?\s*\)\s*_({coord_pat})', r'partial_\1', result)

        # Pattern 1: partial_coord^n followed by variable (partial_x^2 f)
        pattern1 = rf'partial_({coord_pat})\^(\d+)\s+(\w+)'
        result = re.sub(pattern1, r'Derivative(\3, \1, \2)', result)

        # Pattern 2: partial^n_coord followed by variable (partial^2_x f)
        pattern2 = rf'partial\^(\d+)_({coord_pat})\s+(\w+)'
        result = re.sub(pattern2, r'Derivative(\3, \2, \1)', result)

        # Pattern 3: partial_coord followed by variable (partial_x f) - first order
        pattern3 = rf'partial_({coord_pat})\s+(\w+)'
        result = re.sub(pattern3, r'Derivative(\2, \1)', result)

        return result


# =============================================================================
# MAIN PARSER
# =============================================================================

class PDEParser:
    """Parse PDE equations from multiple notation formats.

    The parser detects the format, normalizes to a SymPy-compatible string,
    parses with SymPy, and converts to PDEExpr IR.

    Example:
        >>> parser = PDEParser()

        # Unicode notation
        >>> parsed = parser.parse("-∇²ψ/2 + V(x)ψ = Eψ")

        # LaTeX notation
        >>> parsed = parser.parse(r"-\\frac{\\nabla^2 \\psi}{2} + V(x)\\psi = E\\psi")

        # Subscript notation
        >>> parsed = parser.parse("-psi_xx/2 + V(x)*psi = E*psi")

        # Access the parsed structure
        >>> print(parsed.lhs)  # PDEExpr tree for LHS
        >>> print(parsed.rhs)  # PDEExpr tree for RHS
    """

    # Known unknown functions
    KNOWN_UNKNOWNS = {'u', 'v', 'w', 'f', 'g', 'psi', 'Psi', 'phi', 'Phi', 'chi', 'W', 'h'}

    # Known coordinates
    KNOWN_COORDS = {'x', 'y', 'z', 't', 'r', 'a', 'r2', 'tau', 'eta'}

    # Known parameters
    KNOWN_PARAMS = {'E', 'hbar', 'm', 'theta', 'omega', 'Lambda', 'alpha', 'beta', 'gamma', 'k', 'c'}

    def __init__(self):
        self.format_detector = FormatDetector()
        self.normalizers = {
            'unicode': UnicodeNormalizer(),
            'latex': LatexNormalizer(),
            'typst': TypstNormalizer(),
            'subscript': SubscriptNormalizer(),
            'natural': NaturalNormalizer(),
            'sympy': NaturalNormalizer(),  # SymPy format needs minimal normalization
        }

    def parse(self, equation: str, format: str = "auto") -> ParsedPDE:
        """Parse an equation string to PDEExpr IR.

        Args:
            equation: The equation string in any supported format.
            format: Format hint ('unicode', 'latex', 'typst', 'subscript', 'sympy', 'natural', or 'auto').

        Returns:
            ParsedPDE containing the LHS and RHS expression trees.
        """
        if not SYMPY_AVAILABLE:
            raise ImportError("SymPy is required for parsing. Install with: pip install sympy")

        # Detect format
        if format == "auto":
            format = self.format_detector.detect(equation)

        # Normalize
        normalizer = self.normalizers.get(format, self.normalizers['natural'])
        normalized = normalizer.normalize(equation)

        # Split on =
        lhs_str, rhs_str = self._split_equation(normalized)

        # Build symbol dictionary
        symbols = self._build_symbol_dict(normalized)

        # Parse with SymPy
        try:
            lhs_sympy = self._parse_sympy(lhs_str, symbols)
            rhs_sympy = self._parse_sympy(rhs_str, symbols)
        except Exception as e:
            raise ValueError(f"Failed to parse equation: {equation}\nNormalized: {normalized}\nError: {e}")

        # Convert SymPy → PDEExpr
        coord_names = list(self.KNOWN_COORDS)
        lhs_expr = self._sympy_to_pde_expr(lhs_sympy, coord_names, symbols)
        rhs_expr = self._sympy_to_pde_expr(rhs_sympy, coord_names, symbols)

        return ParsedPDE(
            lhs=lhs_expr,
            rhs=rhs_expr,
            lhs_sympy=lhs_sympy,
            rhs_sympy=rhs_sympy,
            original=equation,
            format=format
        )

    def _split_equation(self, eq: str) -> Tuple[str, str]:
        """Split equation on = sign."""
        if '=' not in eq:
            return eq.strip(), '0'

        parts = eq.split('=')
        if len(parts) != 2:
            raise ValueError(f"Equation must have exactly one '=' sign: {eq}")

        return parts[0].strip(), parts[1].strip()

    def _build_symbol_dict(self, eq: str) -> Dict[str, Any]:
        """Build SymPy symbol dictionary from equation context."""
        symbols = {}

        # First, identify which coordinates are in the equation
        coords_in_eq = []
        for name in self.KNOWN_COORDS:
            if name in eq:
                coords_in_eq.append(name)
                symbols[name] = sp.Symbol(name, real=True)

        # Default coordinate if none found
        if not coords_in_eq:
            coords_in_eq = ['x']
            symbols['x'] = sp.Symbol('x', real=True)

        # Unknown functions - create as applied functions for SymPy parsing
        # This is needed for Derivative(W, r2, 2) to work properly
        for name in self.KNOWN_UNKNOWNS:
            if name in eq:
                # Create the function class
                func = sp.Function(name)
                # Apply it to the coordinates so Derivative works
                coord_symbols = [symbols.get(c, sp.Symbol(c, real=True)) for c in coords_in_eq]
                if len(coord_symbols) == 1:
                    symbols[name] = func(coord_symbols[0])
                else:
                    symbols[name] = func(*coord_symbols)

        # Parameters
        for name in self.KNOWN_PARAMS:
            if name in eq:
                symbols[name] = sp.Symbol(name, real=True, positive=True)

        # Special functions - also apply to coordinates
        coord_symbols = [symbols.get(c, sp.Symbol(c, real=True)) for c in coords_in_eq]
        first_coord = coord_symbols[0] if coord_symbols else sp.Symbol('x', real=True)

        symbols['laplacian'] = sp.Function('laplacian')
        symbols['grad'] = sp.Function('grad')
        symbols['dalembert'] = sp.Function('dalembert')
        symbols['star'] = sp.Function('star')
        symbols['V'] = sp.Function('V')(first_coord)  # V(x) etc.
        symbols['H'] = sp.Function('H')
        symbols['U'] = sp.Function('U')

        # Constants
        symbols['pi'] = sp.pi
        symbols['e'] = sp.E
        symbols['I'] = sp.I

        # SymPy functions
        symbols['Derivative'] = sp.Derivative
        symbols['sin'] = sp.sin
        symbols['cos'] = sp.cos
        symbols['exp'] = sp.exp
        symbols['log'] = sp.log
        symbols['sqrt'] = sp.sqrt

        return symbols

    def _parse_sympy(self, expr_str: str, symbols: Dict[str, Any]) -> Any:
        """Parse a string to SymPy expression."""
        # Handle empty or zero
        expr_str = expr_str.strip()
        if not expr_str or expr_str == '0':
            return sp.Integer(0)

        # Use SymPy parser with implicit multiplication
        transformations = standard_transformations + (implicit_multiplication,)

        try:
            return parse_expr(expr_str, local_dict=symbols, transformations=transformations)
        except Exception:
            # Try with evaluate=False for complex expressions
            return parse_expr(expr_str, local_dict=symbols, transformations=transformations, evaluate=False)

    def _sympy_to_pde_expr(self, expr: Any, coord_names: List[str], symbols: Dict[str, Any]) -> PDEExpr:
        """Convert SymPy expression to PDEExpr tree."""
        if expr is None:
            return Constant(0)

        # Integer/Float
        if isinstance(expr, (sp.Integer, sp.Float, int, float)):
            return Constant(float(expr))

        # Rational
        if isinstance(expr, sp.Rational):
            return Constant(float(expr))

        # Pi, E
        if expr == sp.pi:
            return Constant(3.141592653589793)
        if expr == sp.E:
            return Constant(2.718281828459045)

        # Symbol
        if isinstance(expr, sp.Symbol):
            name = str(expr)
            if name in coord_names or name in self.KNOWN_COORDS:
                return Coordinate(name)
            if name in self.KNOWN_PARAMS:
                return Parameter(name)
            if name in self.KNOWN_UNKNOWNS:
                return Variable(name)
            # Default to parameter
            return Parameter(name)

        # Derivative
        if isinstance(expr, sp.Derivative):
            inner_expr = self._sympy_to_pde_expr(expr.expr, coord_names, symbols)

            # Extract derivative specification
            deriv_vars = []
            for var, count in expr.variable_count:
                deriv_vars.extend([str(var)] * count)

            if len(deriv_vars) == 1:
                return Derivative(inner_expr, deriv_vars[0], 1)
            elif len(set(deriv_vars)) == 1:
                # Same variable repeated
                return Derivative(inner_expr, deriv_vars[0], len(deriv_vars))
            else:
                # Mixed partial
                return Derivative(inner_expr, deriv_vars, 1)

        # Function application
        if isinstance(expr, sp.Function):
            func_name = str(expr.func)
            args = [self._sympy_to_pde_expr(a, coord_names, symbols) for a in expr.args]

            # Special operators
            if func_name == 'laplacian':
                if args:
                    return Laplacian(args[0])
                return Laplacian(Variable('u'))

            if func_name == 'grad':
                if args:
                    return Gradient(args[0])
                return Gradient(Variable('u'))

            if func_name == 'dalembert':
                if args:
                    return Dalembert(args[0])
                return Dalembert(Variable('u'))

            if func_name == 'star':
                if len(args) >= 2:
                    return StarProduct(args[0], args[1])
                return StarProduct(args[0] if args else Variable('f'), Variable('g'))

            # Known math functions
            if func_name.lower() in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'sinh', 'cosh', 'tanh']:
                if args:
                    return UnaryOp(func_name.lower(), args[0])

            # Unknown function (like V(x), H(a))
            if func_name in self.KNOWN_UNKNOWNS:
                return Variable(func_name)

            return FunctionCall(func_name, args)

        # Addition
        if isinstance(expr, sp.Add):
            terms = list(expr.args)
            result = self._sympy_to_pde_expr(terms[0], coord_names, symbols)
            for term in terms[1:]:
                next_term = self._sympy_to_pde_expr(term, coord_names, symbols)
                result = BinaryOp("+", result, next_term)
            return result

        # Multiplication
        if isinstance(expr, sp.Mul):
            factors = list(expr.args)

            # Check for negation: -1 * something
            if factors[0] == -1 and len(factors) == 2:
                return UnaryOp("-", self._sympy_to_pde_expr(factors[1], coord_names, symbols))

            result = self._sympy_to_pde_expr(factors[0], coord_names, symbols)
            for factor in factors[1:]:
                next_factor = self._sympy_to_pde_expr(factor, coord_names, symbols)
                result = BinaryOp("*", result, next_factor)
            return result

        # Power
        if isinstance(expr, sp.Pow):
            base = self._sympy_to_pde_expr(expr.base, coord_names, symbols)
            exp = self._sympy_to_pde_expr(expr.exp, coord_names, symbols)

            # Handle negative exponent (division)
            if isinstance(expr.exp, sp.Integer) and expr.exp < 0:
                return BinaryOp("/", Constant(1), BinaryOp("^", base, Constant(-float(expr.exp))))

            return BinaryOp("^", base, exp)

        # Negation (as -1 * expr)
        if isinstance(expr, sp.Mul) and expr.args[0] == sp.Integer(-1):
            inner = sp.Mul(*expr.args[1:])
            return UnaryOp("-", self._sympy_to_pde_expr(inner, coord_names, symbols))

        # Math functions
        if isinstance(expr, sp.sin):
            return UnaryOp("sin", self._sympy_to_pde_expr(expr.args[0], coord_names, symbols))
        if isinstance(expr, sp.cos):
            return UnaryOp("cos", self._sympy_to_pde_expr(expr.args[0], coord_names, symbols))
        if isinstance(expr, sp.exp):
            return UnaryOp("exp", self._sympy_to_pde_expr(expr.args[0], coord_names, symbols))
        if isinstance(expr, sp.log):
            return UnaryOp("log", self._sympy_to_pde_expr(expr.args[0], coord_names, symbols))
        if isinstance(expr, sp.sqrt):
            return UnaryOp("sqrt", self._sympy_to_pde_expr(expr.args[0], coord_names, symbols))

        # Fallback: try to evaluate as number
        try:
            return Constant(float(expr))
        except (TypeError, ValueError):
            # Last resort: represent as function call
            return FunctionCall(str(expr), [])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse(equation: str, format: str = "auto") -> ParsedPDE:
    """Parse a PDE equation string.

    This is a convenience function that creates a PDEParser and parses
    the equation.

    Args:
        equation: The equation string in any supported format.
        format: Format hint (default: 'auto' for auto-detection).

    Returns:
        ParsedPDE containing the expression trees.

    Example:
        >>> parsed = parse("-∇²ψ/2 + x²ψ/2 = Eψ")
        >>> print(parsed.lhs)
    """
    parser = PDEParser()
    return parser.parse(equation, format)


def detect_format(equation: str) -> str:
    """Detect the notation format of an equation.

    Args:
        equation: The equation string.

    Returns:
        Format name: 'unicode', 'latex', 'subscript', 'sympy', or 'natural'.
    """
    detector = FormatDetector()
    return detector.detect(equation)


# =============================================================================
# EQUATION TEMPLATES
# =============================================================================

TEMPLATES: Dict[str, str] = {
    # Quantum Mechanics
    "schrodinger": "-hbar**2*Derivative(psi, x, 2)/(2*m) + V(x)*psi = E*psi",
    "schrodinger_1d": "-Derivative(psi, x, 2)/2 + V(x)*psi = E*psi",
    "harmonic": "-Derivative(psi, x, 2)/2 + x**2*psi/2 = E*psi",
    "particle_box": "-Derivative(psi, x, 2)/2 = E*psi",
    "hydrogen": "-Derivative(psi, r, 2)/2 - psi/r = E*psi",

    # Quantum Cosmology
    "wheeler_dewitt": "-hbar**2*Derivative(Psi, a, 2) + U(a)*Psi = 0",
    "wheeler_dewitt_simple": "-Derivative(Psi, a, 2) + U(a)*Psi = 0",
    "deformed_wdw": "star(H, Psi, theta) = 0",

    # Field Theory
    "klein_gordon": "Derivative(phi, t, 2) - Derivative(phi, x, 2) + m**2*phi = 0",
    "klein_gordon_1d": "Derivative(phi, t, 2) - c**2*Derivative(phi, x, 2) + m**2*phi = 0",
    "wave": "Derivative(u, t, 2) = c**2*Derivative(u, x, 2)",
    "wave_1d": "Derivative(u, t, 2) - Derivative(u, x, 2) = 0",

    # Classical PDEs
    "heat": "Derivative(u, t) = alpha*Derivative(u, x, 2)",
    "heat_1d": "Derivative(u, t) = Derivative(u, x, 2)",
    "laplace": "Derivative(u, x, 2) + Derivative(u, y, 2) = 0",
    "laplace_1d": "Derivative(u, x, 2) = 0",
    "poisson": "Derivative(u, x, 2) + Derivative(u, y, 2) = f(x, y)",
    "poisson_1d": "Derivative(u, x, 2) = f(x)",

    # Cosmology
    "friedmann": "H**2 = 8*pi*G*rho/3 - k/a**2 + Lambda/3",

    # Wigner function
    "wigner_ho": "(r2 - 2*E)*W - (hbar**2/4)*(4*r2*Derivative(W, r2, 2) + 4*Derivative(W, r2)) = 0",
}


def get_template(name: str) -> str:
    """Get an equation template by name.

    Args:
        name: Template name (e.g., 'schrodinger', 'wave', 'heat').

    Returns:
        Equation string.

    Raises:
        KeyError: If template not found.
    """
    name_lower = name.lower().replace('-', '_').replace(' ', '_')
    if name_lower not in TEMPLATES:
        available = ', '.join(sorted(TEMPLATES.keys()))
        raise KeyError(f"Unknown template: {name}. Available: {available}")
    return TEMPLATES[name_lower]


def list_templates() -> List[str]:
    """List all available equation templates."""
    return sorted(TEMPLATES.keys())
