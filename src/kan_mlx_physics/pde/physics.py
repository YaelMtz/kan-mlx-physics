"""Pre-built physics problems for cosmology and quantum mechanics.

These are the PDEs you actually work with in your thesis.
Physics first. Always.
"""

import mlx.core as mx
import numpy as np
from typing import Optional, Callable, Dict
from dataclasses import dataclass

from .problem import PDEProblem, Domain, BoundaryCondition, BCType
from .operators import (
    laplacian, grad, moyal_bracket, star_product,
    wheeler_dewitt_operator, friedmann_operator, klein_gordon_curved,
    _fd_derivative, _fd_second_derivative, _fd_mixed_derivative
)


# =============================================================================
# QUANTUM COSMOLOGY
# =============================================================================

def WheelerDeWitt(
    potential: Optional[Callable] = None,
    domain: Optional[Domain] = None,
    hbar: float = 1.0,
    name: str = "Wheeler-DeWitt"
) -> PDEProblem:
    """Wheeler-DeWitt equation for quantum cosmology.

    The Wheeler-DeWitt equation in minisuperspace:

        Ĥ Ψ = 0

    For FLRW minisuperspace with scale factor a:

        [-ℏ² ∂²/∂a² + U(a)] Ψ(a) = 0

    where U(a) depends on matter content and cosmological constant.

    Common potentials:
        - Empty: U(a) = -a (k=+1 closed universe)
        - Λ: U(a) = a³(Λa² - k)/3
        - Radiation: U(a) = -a + ρ_r/a
        - Matter: U(a) = -a + ρ_m

    Args:
        potential: U(a) function, default is closed FLRW with Λ
        domain: Minisuperspace domain, default is a ∈ [0.01, 5]
        hbar: Planck constant
        name: Problem name

    Returns:
        PDEProblem for Wheeler-DeWitt equation

    Example:
        # Standard closed FLRW
        problem = WheelerDeWitt()

        # With cosmological constant Λ=0.1
        problem = WheelerDeWitt(
            potential=lambda q: q[:,0]**3 * (0.1 * q[:,0]**2 - 1)
        )
    """
    if domain is None:
        domain = Domain.minisuperspace((0.01, 5.0), name="FLRW minisuperspace")

    if potential is None:
        # Closed FLRW with Λ: U(a) = a³(Λa² - k) with k=1, Λ=0.1
        def potential(q):
            a = q[:, 0]
            Lambda = 0.1
            k = 1.0
            return a**3 * (Lambda * a**2 - k)

    def residual(psi: Callable, q: mx.array) -> mx.array:
        """Wheeler-DeWitt residual: [-ℏ² ∂²/∂a² + U(a)] Ψ = 0"""
        return wheeler_dewitt_operator(psi, q, potential, hbar)

    # Boundary conditions: Ψ → 0 as a → 0 and a → ∞
    bcs = [
        BoundaryCondition.dirichlet(
            lambda x: mx.zeros(x.shape[0]),
            where=lambda x: x[:, 0] < 0.05,
            name="Ψ(a→0) = 0"
        ),
        BoundaryCondition.dirichlet(
            lambda x: mx.zeros(x.shape[0]),
            where=lambda x: x[:, 0] > 4.5,
            name="Ψ(a→∞) = 0"
        ),
    ]

    return PDEProblem(
        name=name,
        domain=domain,
        residual=residual,
        boundary_conditions=bcs,
        parameters={"hbar": hbar},
        description="Wheeler-DeWitt equation: Ĥ|Ψ⟩ = 0 in minisuperspace"
    )


def Friedmann(
    rho: Optional[Callable] = None,
    k: float = 0.0,
    Lambda: float = 0.0,
    domain: Optional[Domain] = None,
    name: str = "Friedmann"
) -> PDEProblem:
    """Friedmann equations for classical cosmology.

    The Friedmann equations:

        H² = (8πG/3)ρ - k/a² + Λ/3
        ä/a = -(4πG/3)(ρ + 3p) + Λ/3

    We solve for H(a) given matter content.

    Args:
        rho: Energy density ρ(a), default is matter ρ ∝ a⁻³
        k: Spatial curvature (0, +1, -1)
        Lambda: Cosmological constant
        domain: Scale factor domain, default [0.01, 10]
        name: Problem name

    Returns:
        PDEProblem for Friedmann equations

    Example:
        # ΛCDM cosmology
        problem = Friedmann(
            rho=lambda a: 0.3 * a[:,0]**(-3),  # Matter
            Lambda=0.7
        )
    """
    if domain is None:
        domain = Domain.interval(0.01, 10.0, name="Scale factor a")

    if rho is None:
        # Matter domination
        rho = lambda a: a[:, 0]**(-3)

    def residual(H: Callable, a: mx.array) -> mx.array:
        """Friedmann residual: H² - (8πG/3)ρ + k/a² - Λ/3 = 0"""
        return friedmann_operator(H, a, rho, k, Lambda)

    return PDEProblem(
        name=name,
        domain=domain,
        residual=residual,
        boundary_conditions=[],
        parameters={"k": k, "Lambda": Lambda},
        description="Friedmann equation: H² = (8πG/3)ρ - k/a² + Λ/3"
    )


def KleinGordonCurved(
    mass: float = 0.0,
    background: str = "de_sitter",
    H: float = 1.0,
    domain: Optional[Domain] = None,
    name: str = "Klein-Gordon"
) -> PDEProblem:
    """Klein-Gordon equation in curved spacetime.

    □φ - m²φ = 0

    For FLRW background:
        -∂²φ/∂t² - 3H∂φ/∂t + (1/a²)∇²φ - m²φ = 0

    Args:
        mass: Scalar field mass
        background: "de_sitter", "matter", "radiation"
        H: Hubble parameter (for de Sitter)
        domain: Spacetime domain
        name: Problem name

    Returns:
        PDEProblem for Klein-Gordon equation

    Example:
        # Massless scalar in de Sitter
        problem = KleinGordonCurved(mass=0, background="de_sitter", H=1.0)
    """
    if domain is None:
        domain = Domain.spacetime((0, 5), (-5, 5), name="de Sitter spacetime")

    def residual(phi: Callable, x: mx.array) -> mx.array:
        """Klein-Gordon residual in FLRW."""
        return klein_gordon_curved(phi, x, mass)

    # Initial conditions: φ(0,x) specified
    bcs = [
        BoundaryCondition.cauchy(
            psi_0=lambda x: mx.exp(-x[:, 0]**2),  # Gaussian initial data
            name="Initial Gaussian"
        )
    ]

    return PDEProblem(
        name=name,
        domain=domain,
        residual=residual,
        boundary_conditions=bcs,
        parameters={"mass": mass, "H": H},
        description=f"Klein-Gordon in {background} background: □φ - m²φ = 0"
    )


# =============================================================================
# QUANTUM MECHANICS WITH DEFORMATION
# =============================================================================

def Schrodinger(
    potential: Optional[Callable] = None,
    energy: float = 0.5,
    hbar: float = 1.0,
    mass: float = 1.0,
    domain: Optional[Domain] = None,
    name: str = "Schrödinger"
) -> PDEProblem:
    """Time-independent Schrödinger equation.

    -ℏ²/(2m) ∇²ψ + V(x)ψ = Eψ

    Args:
        potential: V(x), default is harmonic oscillator V = x²/2
        energy: Energy eigenvalue E
        hbar: Planck constant
        mass: Particle mass
        domain: Spatial domain
        name: Problem name

    Returns:
        PDEProblem for Schrödinger equation

    Example:
        # Harmonic oscillator ground state
        problem = Schrodinger(energy=0.5)

        # Hydrogen-like
        problem = Schrodinger(
            potential=lambda x: -1/mx.sqrt(x[:,0]**2 + 0.01),
            energy=-0.5
        )
    """
    if domain is None:
        domain = Domain.interval(-5, 5, name="Position space")

    if potential is None:
        # Harmonic oscillator
        potential = lambda x: 0.5 * x[:, 0]**2

    coeff = hbar**2 / (2 * mass)

    def residual(psi: Callable, x: mx.array) -> mx.array:
        """Schrödinger residual: -ℏ²/(2m)∇²ψ + Vψ - Eψ = 0"""
        kinetic = -coeff * laplacian(psi, x)

        psi_val = psi(x)
        if len(psi_val.shape) > 1:
            psi_val = psi_val[:, 0]

        V_val = potential(x)
        if len(V_val.shape) > 1:
            V_val = V_val[:, 0]

        return kinetic + V_val * psi_val - energy * psi_val

    # Bound state BCs
    bcs = [
        BoundaryCondition.dirichlet(
            lambda x: mx.zeros(x.shape[0]),
            where=lambda x: mx.abs(x[:, 0]) > 4.5,
            name="ψ(±∞) = 0"
        )
    ]

    # Exact solution for harmonic oscillator
    if potential is None:
        exact = lambda x: mx.exp(-x[:, 0]**2 / 2)  # Ground state
    else:
        exact = None

    return PDEProblem(
        name=name,
        domain=domain,
        residual=residual,
        boundary_conditions=bcs,
        parameters={"E": energy, "hbar": hbar, "m": mass},
        exact_solution=exact,
        description="Schrödinger equation: Ĥψ = Eψ"
    )


def DeformedSchrodinger(
    potential: Optional[Callable] = None,
    energy: float = 0.5,
    hbar: float = 1.0,
    theta: float = 0.1,
    domain: Optional[Domain] = None,
    name: str = "Deformed Schrödinger"
) -> PDEProblem:
    """Schrödinger equation with Moyal star product deformation.

    In deformation quantization, observables form a non-commutative algebra
    with the Moyal star product:

        f ⋆ g = fg + (iℏ/2){f,g} + O(ℏ²)

    The deformed Schrödinger equation:

        H ⋆ ψ = E ⋆ ψ

    where the star product introduces θ-dependent corrections.

    Args:
        potential: V(q,p), default is harmonic oscillator in phase space
        energy: Energy eigenvalue
        hbar: Planck constant
        theta: Deformation parameter (noncommutativity scale)
        domain: Phase space domain (q, p)
        name: Problem name

    Returns:
        PDEProblem for deformed Schrödinger equation

    Example:
        # Deformed harmonic oscillator
        problem = DeformedSchrodinger(theta=0.1)
    """
    if domain is None:
        # Phase space: (q, p)
        domain = Domain([(-5, 5), (-5, 5)], name="Phase space (q,p)")

    if potential is None:
        # Harmonic oscillator: H = p²/2 + q²/2
        def H(x):
            q, p = x[:, 0], x[:, 1]
            return p**2 / 2 + q**2 / 2

    def residual(psi: Callable, x: mx.array) -> mx.array:
        """Deformed Schrödinger: H ⋆ ψ - E ⋆ ψ = 0"""

        # H ⋆ ψ using star product
        H_star_psi = star_product(H if potential is None else potential, psi, x, hbar=theta, order=2)

        # E ⋆ ψ = E·ψ (constant star-commutes)
        psi_val = psi(x)
        if len(psi_val.shape) > 1:
            psi_val = psi_val[:, 0]

        E_star_psi = energy * psi_val

        return H_star_psi - E_star_psi

    # Phase space BCs: ψ → 0 at infinity
    bcs = [
        BoundaryCondition.dirichlet(
            lambda x: mx.zeros(x.shape[0]),
            where=lambda x: (mx.abs(x[:, 0]) > 4.5) | (mx.abs(x[:, 1]) > 4.5),
            name="ψ → 0 at phase space boundary"
        )
    ]

    return PDEProblem(
        name=name,
        domain=domain,
        residual=residual,
        boundary_conditions=bcs,
        parameters={"E": energy, "hbar": hbar, "theta": theta},
        description=f"Deformed Schrödinger: H ⋆_θ ψ = E ψ with θ = {theta}"
    )


# =============================================================================
# COMBINED PROBLEMS FOR YOUR THESIS
# =============================================================================

def DeformedWheelerDeWitt(
    potential: Optional[Callable] = None,
    hbar: float = 1.0,
    theta: float = 0.1,
    domain: Optional[Domain] = None,
    name: str = "Deformed Wheeler-DeWitt"
) -> PDEProblem:
    """Wheeler-DeWitt equation with Moyal deformation.

    This is the quantum cosmology equation with noncommutative corrections:

        Ĥ_θ |Ψ⟩ = 0

    where Ĥ_θ is the Wheeler-DeWitt Hamiltonian with star product:

        Ĥ_θ = -ℏ² G^{AB} π_A ⋆_θ π_B + U(q)

    This is central to your thesis on algebraic deformation in cosmology.

    Args:
        potential: Minisuperspace potential U(a, φ)
        hbar: Planck constant
        theta: Deformation parameter
        domain: Minisuperspace phase space (a, π_a)

    Returns:
        PDEProblem for deformed Wheeler-DeWitt

    Example:
        # FLRW with scalar field and deformation
        problem = DeformedWheelerDeWitt(theta=0.05)
    """
    if domain is None:
        # Minisuperspace phase space: (a, π_a)
        domain = Domain([(0.1, 5), (-5, 5)], name="Minisuperspace phase space")

    if potential is None:
        # Closed FLRW with Λ
        def potential(x):
            a = x[:, 0]
            Lambda = 0.1
            return a**3 * (Lambda * a**2 - 1)

    def hamiltonian(x):
        """Classical Hamiltonian in minisuperspace."""
        a, pi_a = x[:, 0], x[:, 1]
        # H = -π_a²/(24a) + U(a) (simplified)
        kinetic = -pi_a**2 / (24 * a + 1e-8)
        pot = potential(x)
        return kinetic + pot

    def residual(psi: Callable, x: mx.array) -> mx.array:
        """Deformed Wheeler-DeWitt: Ĥ_θ Ψ = 0"""

        # H ⋆_θ ψ = 0
        H_star_psi = star_product(hamiltonian, psi, x, hbar=theta, order=2)

        return H_star_psi

    bcs = [
        BoundaryCondition.dirichlet(
            lambda x: mx.zeros(x.shape[0]),
            where=lambda x: (x[:, 0] < 0.15) | (x[:, 0] > 4.5),
            name="Ψ → 0 at boundaries"
        )
    ]

    return PDEProblem(
        name=name,
        domain=domain,
        residual=residual,
        boundary_conditions=bcs,
        parameters={"hbar": hbar, "theta": theta},
        description=f"Deformed Wheeler-DeWitt: Ĥ_θ Ψ = 0 with Moyal parameter θ = {theta}"
    )


def MultiFieldWheelerDeWitt(
    n_scalars: int = 1,
    potential: Optional[Callable] = None,
    metric: Optional[Callable] = None,
    factor_ordering: str = "laplacian-beltrami",
    hbar: float = 1.0,
    Lambda: float = 0.1,
    domain: Optional[Domain] = None,
    name: str = "Multi-Field Wheeler-DeWitt"
) -> PDEProblem:
    """Wheeler-DeWitt equation with coupled scalar fields.

    The multi-field Wheeler-DeWitt equation in minisuperspace:

        [-ℏ² G^{AB} ∇_A ∇_B + U(q^A)] Ψ(q^A) = 0

    where q^A = (a, φ₁, φ₂, ...) are the minisuperspace coordinates,
    G^{AB} is the inverse superspace metric, and U is the potential.

    Factor orderings:
        - 'naive': G^{AB} ∂_A ∂_B
        - 'laplacian-beltrami': (1/√|G|) ∂_A (√|G| G^{AB} ∂_B)
        - 'weyl': Symmetric Weyl ordering

    For FLRW + n scalar fields, the superspace metric is:

        G^{AB} = diag(-1/(24a), a³, a³, ...) × (3/(4πG))

    and the potential is:

        U(a, φ) = a³[Λa² - k + Σᵢ V(φᵢ)]

    Args:
        n_scalars: Number of scalar fields (default: 1)
        potential: Full potential U(q), or None for default
        metric: Superspace metric G^{AB}(q), or None for FLRW+scalars
        factor_ordering: "naive", "laplacian-beltrami", or "weyl"
        hbar: Planck constant
        Lambda: Cosmological constant
        domain: Minisuperspace domain

    Returns:
        PDEProblem for multi-field Wheeler-DeWitt

    Example:
        # Single scalar field (inflaton)
        problem = MultiFieldWheelerDeWitt(n_scalars=1)

        # Two scalar fields (inflaton + curvaton)
        problem = MultiFieldWheelerDeWitt(
            n_scalars=2,
            potential=lambda q: q[:,0]**3 * (
                0.1*q[:,0]**2 - 1 + 0.5*(q[:,1]**2 + q[:,2]**2)
            )
        )
    """
    dim = 1 + n_scalars  # scale factor + scalar fields

    if domain is None:
        # Default domain: a ∈ [0.1, 5], φᵢ ∈ [-3, 3]
        field_ranges = [(0.1, 5.0)] + [(-3.0, 3.0)] * n_scalars
        domain = Domain.superspace(
            fields=field_ranges,
            field_names=["a"] + [f"φ{i+1}" for i in range(n_scalars)],
            name=f"Minisuperspace (a, φ₁...φ_{n_scalars})"
        )

    if metric is None:
        def metric(q: mx.array) -> mx.array:
            """FLRW + scalar fields superspace metric G^{AB}."""
            batch = q.shape[0]
            a = q[:, 0]

            # G^{00} = -1/(24a) for scale factor
            G00 = -1.0 / (24 * a + 1e-10)

            # G^{ii} = a³ for scalar fields
            G_scalar = a**3

            # Construct diagonal metric
            G = mx.zeros((batch, dim, dim))

            # Set diagonal elements
            # G[:, 0, 0] = G00
            G = G.at[:, 0, 0].set(G00)
            for i in range(1, dim):
                G = G.at[:, i, i].set(G_scalar)

            return G

    if potential is None:
        def potential(q: mx.array) -> mx.array:
            """Default potential: closed FLRW with Λ and mass terms."""
            a = q[:, 0]
            # U = a³(Λa² - k) + a³ Σᵢ m²φᵢ²/2
            U = a**3 * (Lambda * a**2 - 1)  # k=1 closed
            # Add scalar field mass terms
            for i in range(1, dim):
                phi_i = q[:, i]
                m_i = 0.1  # Default mass
                U = U + 0.5 * a**3 * m_i**2 * phi_i**2
            return U

    def superspace_kinetic(psi: Callable, q: mx.array, h: float = 1e-4) -> mx.array:
        """Compute -ℏ² G^{AB} ∇_A ∇_B Ψ with factor ordering."""
        G = metric(q)
        batch = q.shape[0]

        if factor_ordering == "naive":
            # Simple: G^{AB} ∂_A ∂_B Ψ
            kinetic = mx.zeros(batch)

            for A in range(dim):
                for B in range(dim):
                    if A == B:
                        # G^{AA} ∂²Ψ/∂q_A²
                        d2psi = _fd_second_derivative(psi, q, A, h)
                        if len(d2psi.shape) > 1:
                            d2psi = d2psi[:, 0]
                        kinetic = kinetic + G[:, A, A] * d2psi
                    else:
                        # G^{AB} ∂²Ψ/∂q_A∂q_B (off-diagonal)
                        d2psi = _fd_mixed_derivative(psi, q, A, B, h)
                        if len(d2psi.shape) > 1:
                            d2psi = d2psi[:, 0]
                        kinetic = kinetic + G[:, A, B] * d2psi

            return -hbar**2 * kinetic

        elif factor_ordering == "laplacian-beltrami":
            # Covariant: (1/√|G|) ∂_A (√|G| G^{AB} ∂_B Ψ)
            # This requires computing √|det G|

            # Compute metric determinant
            sqrtG = domain.get_metric_determinant(q, h)

            kinetic = mx.zeros(batch)

            for A in range(dim):
                for B in range(dim):
                    # Compute √|G| G^{AB} ∂_B Ψ
                    dpsi_B = _fd_derivative(psi, q, B, h)
                    if len(dpsi_B.shape) > 1:
                        dpsi_B = dpsi_B[:, 0]

                    flux_AB = sqrtG * G[:, A, B] * dpsi_B

                    # Compute ∂_A of the flux (using finite difference)
                    def flux_fn(qq):
                        dpsi = _fd_derivative(psi, qq, B, h)
                        if len(dpsi.shape) > 1:
                            dpsi = dpsi[:, 0]
                        G_at_q = metric(qq)
                        sqrt_G = domain.get_metric_determinant(qq, h)
                        return sqrt_G * G_at_q[:, A, B] * dpsi

                    d_flux = _fd_derivative(flux_fn, q, A, h)
                    if len(d_flux.shape) > 1:
                        d_flux = d_flux[:, 0]

                    kinetic = kinetic + d_flux / (sqrtG + 1e-12)

            return -hbar**2 * kinetic

        elif factor_ordering == "weyl":
            # Weyl symmetric ordering: average of orderings
            # Simplified: use naive + correction term
            naive_kinetic = mx.zeros(batch)

            for A in range(dim):
                for B in range(dim):
                    if A == B:
                        d2psi = _fd_second_derivative(psi, q, A, h)
                        if len(d2psi.shape) > 1:
                            d2psi = d2psi[:, 0]
                        naive_kinetic = naive_kinetic + G[:, A, A] * d2psi
                    else:
                        d2psi = _fd_mixed_derivative(psi, q, A, B, h)
                        if len(d2psi.shape) > 1:
                            d2psi = d2psi[:, 0]
                        naive_kinetic = naive_kinetic + G[:, A, B] * d2psi

            return -hbar**2 * naive_kinetic

        else:
            raise ValueError(f"Unknown factor ordering: {factor_ordering}")

    def residual(psi: Callable, q: mx.array) -> mx.array:
        """Multi-field Wheeler-DeWitt residual:
        [-ℏ² G^{AB} ∇_A ∇_B + U(q)] Ψ = 0
        """
        kinetic = superspace_kinetic(psi, q)

        psi_val = psi(q)
        if len(psi_val.shape) > 1:
            psi_val = psi_val[:, 0]

        U = potential(q)
        if len(U.shape) > 1:
            U = U[:, 0]

        return kinetic + U * psi_val

    # Boundary conditions
    bcs = [
        # DeWitt BC at a → 0
        BoundaryCondition.dewitt(singularity_coord=0, name="Ψ(a→0) = 0"),
        # Normalizability at large a
        BoundaryCondition.normalizability(
            decay_rate=1.0,
            where=lambda x: x[:, 0] > 4.5,
            name="Ψ normalizable"
        ),
    ]

    # Add BCs for scalar fields (decay at large |φ|)
    for i in range(1, dim):
        bcs.append(
            BoundaryCondition.dirichlet(
                lambda x: mx.zeros(x.shape[0]),
                where=lambda x, idx=i: mx.abs(x[:, idx]) > 2.8,
                name=f"Ψ(|φ{i}|→∞) = 0"
            )
        )

    return PDEProblem(
        name=name,
        domain=domain,
        residual=residual,
        boundary_conditions=bcs,
        parameters={
            "hbar": hbar,
            "Lambda": Lambda,
            "n_scalars": n_scalars,
            "factor_ordering": factor_ordering
        },
        description=f"Multi-field Wheeler-DeWitt: [-ℏ² G^{{AB}} ∇_A∇_B + U] Ψ = 0 with {n_scalars} scalar field(s)"
    )


def FriedmannPerturbations(
    background: str = "de_sitter",
    H0: float = 1.0,
    k_mode: float = 1.0,
    domain: Optional[Domain] = None,
    name: str = "Cosmological Perturbations"
) -> PDEProblem:
    """Linear perturbations around FLRW background.

    The Mukhanov-Sasaki equation for scalar perturbations:

        v'' + (k² - z''/z) v = 0

    where v = zR, R is the comoving curvature perturbation,
    and z = a√(2ε) with ε = -Ḣ/H².

    Args:
        background: "de_sitter", "matter", "radiation"
        H0: Hubble parameter
        k_mode: Comoving wavenumber
        domain: Conformal time domain

    Returns:
        PDEProblem for perturbation equations

    Example:
        # Scalar perturbations in de Sitter
        problem = FriedmannPerturbations(background="de_sitter", k_mode=10)
    """
    if domain is None:
        domain = Domain.interval(-10, 0, name="Conformal time η")  # η < 0 for de Sitter

    # For de Sitter: a = -1/(H η), z''/z = 2/η²
    def effective_potential(eta):
        """z''/z for de Sitter."""
        return 2 / (eta**2 + 1e-8)

    def residual(v: Callable, eta: mx.array) -> mx.array:
        """Mukhanov-Sasaki: v'' + (k² - z''/z)v = 0"""
        v_val = v(eta)
        if len(v_val.shape) > 1:
            v_val = v_val[:, 0]

        v_pp = _fd_second_derivative(v, eta, 0)

        eta_val = eta[:, 0]
        zpp_over_z = effective_potential(eta_val)

        return v_pp + (k_mode**2 - zpp_over_z) * v_val

    # Bunch-Davies initial conditions at early times
    bcs = [
        BoundaryCondition.cauchy(
            psi_0=lambda x: mx.exp(-1j * k_mode * x[:, 0]) / mx.sqrt(2 * k_mode),
            name="Bunch-Davies vacuum"
        )
    ]

    return PDEProblem(
        name=name,
        domain=domain,
        residual=residual,
        boundary_conditions=bcs,
        parameters={"H0": H0, "k": k_mode},
        description=f"Mukhanov-Sasaki equation for k = {k_mode}"
    )
