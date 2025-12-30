"""PDE problem definition with physics-first design.

Elegantly define PDEs through their physics, not their numerics.
"""

import mlx.core as mx
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Union, Tuple, TYPE_CHECKING
from enum import Enum, auto

if TYPE_CHECKING:
    pass


class BCType(Enum):
    """Boundary condition types."""
    DIRICHLET = auto()    # φ = f on boundary
    NEUMANN = auto()      # ∂φ/∂n = g on boundary
    ROBIN = auto()        # αφ + β∂φ/∂n = g
    PERIODIC = auto()     # φ(a) = φ(b)
    CAUCHY = auto()       # φ and ∂φ/∂t at t=0
    SOMMERFELD = auto()   # Radiation condition (outgoing waves)
    # Quantum cosmology specific
    WKB = auto()          # WKB matching: Ψ ~ exp(±iS/ℏ)
    DEWITT = auto()       # DeWitt BC: Ψ(a=0) = 0 (singularity avoidance)
    NORMALIZABILITY = auto()  # ∫|Ψ|² < ∞ with asymptotic decay


@dataclass
class BoundaryCondition:
    """A boundary condition on the domain.

    Examples:
        # Dirichlet: ψ(0) = 0
        BoundaryCondition.dirichlet(lambda x: 0, where=lambda x: x[:,0] == 0)

        # Neumann: ∂ψ/∂x = 0 at x=L
        BoundaryCondition.neumann(lambda x: 0, where=lambda x: x[:,0] == L)

        # Periodic
        BoundaryCondition.periodic(dim=0, period=2*pi)

        # Cauchy (initial conditions)
        BoundaryCondition.cauchy(psi_0=gaussian, dpsi_0=zero)
    """
    bc_type: BCType
    value: Optional[Callable] = None          # Boundary value function
    where: Optional[Callable] = None          # Condition for boundary points
    dim: int = 0                               # Dimension for periodic BC
    period: float = 1.0                        # Period for periodic BC
    alpha: float = 1.0                         # Robin BC coefficient
    beta: float = 1.0                          # Robin BC coefficient
    name: str = ""                             # Human-readable name

    @classmethod
    def dirichlet(cls, value: Callable, where: Callable, name: str = "Dirichlet"):
        """Dirichlet BC: φ = value on boundary."""
        return cls(BCType.DIRICHLET, value=value, where=where, name=name)

    @classmethod
    def neumann(cls, value: Callable, where: Callable, name: str = "Neumann"):
        """Neumann BC: ∂φ/∂n = value on boundary."""
        return cls(BCType.NEUMANN, value=value, where=where, name=name)

    @classmethod
    def periodic(cls, dim: int = 0, period: float = 2 * np.pi, name: str = "Periodic"):
        """Periodic BC: φ(x) = φ(x + period)."""
        return cls(BCType.PERIODIC, dim=dim, period=period, name=name)

    @classmethod
    def cauchy(cls, psi_0: Callable, dpsi_0: Optional[Callable] = None, name: str = "Cauchy"):
        """Cauchy/initial conditions: φ(0,x) = ψ₀(x), ∂φ/∂t(0,x) = ψ̇₀(x)."""
        return cls(BCType.CAUCHY, value=psi_0,
                   where=dpsi_0,  # Repurpose 'where' for velocity IC
                   name=name)

    @classmethod
    def sommerfeld(cls, wave_speed: float = 1.0, name: str = "Sommerfeld"):
        """Radiation BC for outgoing waves."""
        return cls(BCType.SOMMERFELD, alpha=wave_speed, name=name)

    @classmethod
    def robin(cls, alpha: float, beta: float, value: Callable = None,
              where: Callable = None, name: str = "Robin"):
        """Robin BC: αΨ + β∂Ψ/∂n = value.

        Args:
            alpha: Coefficient of Ψ
            beta: Coefficient of ∂Ψ/∂n (normal derivative)
            value: RHS function (default: 0)
            where: Condition for boundary points

        Example:
            # Mixed BC: Ψ + 2∂Ψ/∂n = 0
            BoundaryCondition.robin(alpha=1.0, beta=2.0, where=lambda x: x[:,0] > 4.9)
        """
        if value is None:
            value = lambda x: mx.zeros(x.shape[0])
        return cls(BCType.ROBIN, value=value, where=where, alpha=alpha, beta=beta, name=name)

    @classmethod
    def wkb_matching(cls, S: Callable, sign: int = 1, hbar: float = 1.0,
                     where: Callable = None, name: str = "WKB"):
        """WKB matching BC: Ψ ~ exp(±iS/ℏ) at classical turning points.

        For quantum cosmology, matches to semiclassical WKB solution
        in the classically allowed region.

        Args:
            S: Classical action function S(x)
            sign: +1 for outgoing wave, -1 for incoming wave
            hbar: Planck constant
            where: Condition for WKB matching region

        Example:
            # Match to outgoing WKB in tunneling region
            BoundaryCondition.wkb_matching(
                S=lambda x: x[:,0]**2,  # Quadratic action
                sign=+1,
                hbar=0.05,
                where=lambda x: x[:,0] > 3.0
            )
        """
        bc = cls(BCType.WKB, value=S, where=where, alpha=float(sign), beta=hbar, name=name)
        return bc

    @classmethod
    def dewitt(cls, singularity_coord: int = 0, singularity_value: float = 0.0,
               name: str = "DeWitt"):
        """DeWitt BC: Ψ(a=0) = 0 for singularity avoidance.

        In quantum cosmology, this BC ensures the wavefunction
        vanishes at the initial singularity (a=0).

        Args:
            singularity_coord: Which coordinate has the singularity (default: 0 for scale factor)
            singularity_value: Value at singularity (default: 0)

        Example:
            # Require Ψ = 0 at a → 0
            BoundaryCondition.dewitt(singularity_coord=0)
        """
        where = lambda x: x[:, singularity_coord] < singularity_value + 0.1
        value = lambda x: mx.zeros(x.shape[0])
        return cls(BCType.DEWITT, value=value, where=where, dim=singularity_coord, name=name)

    @classmethod
    def normalizability(cls, decay_rate: float = 1.0, decay_type: str = "exponential",
                        where: Callable = None, name: str = "Normalizability"):
        """Normalizability BC: |Ψ| ~ exp(-decay_rate·r) at infinity.

        Ensures ∫|Ψ|² dx < ∞ by enforcing exponential decay at large distances.

        Args:
            decay_rate: Rate of exponential decay
            decay_type: "exponential" or "gaussian"
            where: Region where decay is enforced

        Example:
            # Exponential decay at large scale factor
            BoundaryCondition.normalizability(
                decay_rate=1.0,
                where=lambda x: x[:,0] > 4.5
            )
        """
        bc = cls(BCType.NORMALIZABILITY, where=where, alpha=decay_rate, name=name)
        bc._decay_type = decay_type
        return bc


@dataclass
class Domain:
    """Physical domain for the PDE.

    Supports:
    - Rectangular domains in n dimensions
    - Spherical/radial coordinates
    - Custom sampling strategies
    """
    bounds: List[Tuple[float, float]]       # [(x_min, x_max), (y_min, y_max), ...]
    coord_type: str = "cartesian"            # "cartesian", "spherical", "cylindrical"
    name: str = ""

    @property
    def dim(self) -> int:
        return len(self.bounds)

    def sample(self, n: int, strategy: str = "uniform", seed: int = 42) -> mx.array:
        """Sample points from the domain.

        Args:
            n: Number of points
            strategy: "uniform", "sobol", "boundary", "adaptive"
            seed: Random seed

        Returns:
            Points (n, dim)
        """
        np.random.seed(seed)

        if strategy == "uniform":
            points = []
            for lo, hi in self.bounds:
                points.append(np.random.uniform(lo, hi, n))
            return mx.array(np.stack(points, axis=-1).astype(np.float32))

        elif strategy == "grid":
            n_per_dim = int(n ** (1.0 / self.dim))
            grids = [np.linspace(lo, hi, n_per_dim) for lo, hi in self.bounds]
            mesh = np.meshgrid(*grids, indexing='ij')
            points = np.stack([m.flatten() for m in mesh], axis=-1)
            return mx.array(points.astype(np.float32))

        elif strategy == "boundary":
            # Sample more densely near boundaries
            points = []
            for lo, hi in self.bounds:
                # Beta distribution concentrates near 0 and 1
                u = np.random.beta(0.5, 0.5, n)
                points.append(lo + (hi - lo) * u)
            return mx.array(np.stack(points, axis=-1).astype(np.float32))

        elif strategy == "sobol":
            # Quasi-random Sobol sequence for better coverage
            try:
                from scipy.stats import qmc
                sampler = qmc.Sobol(d=self.dim, seed=seed)
                u = sampler.random(n)
                points = []
                for i, (lo, hi) in enumerate(self.bounds):
                    points.append(lo + (hi - lo) * u[:, i])
                return mx.array(np.stack(points, axis=-1).astype(np.float32))
            except ImportError:
                return self.sample(n, "uniform", seed)

        else:
            return self.sample(n, "uniform", seed)

    def sample_boundary(self, n: int, seed: int = 42) -> mx.array:
        """Sample points on the domain boundary."""
        np.random.seed(seed)
        points = []

        # For each face of the hypercube
        for d in range(self.dim):
            for boundary_val in [self.bounds[d][0], self.bounds[d][1]]:
                n_face = n // (2 * self.dim)
                face_points = []
                for i, (lo, hi) in enumerate(self.bounds):
                    if i == d:
                        face_points.append(np.full(n_face, boundary_val))
                    else:
                        face_points.append(np.random.uniform(lo, hi, n_face))
                points.append(np.stack(face_points, axis=-1))

        return mx.array(np.concatenate(points, axis=0).astype(np.float32))

    @classmethod
    def interval(cls, a: float, b: float, name: str = ""):
        """1D interval [a, b]."""
        return cls([(a, b)], name=name or f"[{a}, {b}]")

    @classmethod
    def rectangle(cls, x_range: Tuple[float, float], y_range: Tuple[float, float], name: str = ""):
        """2D rectangle."""
        return cls([x_range, y_range], name=name or "Rectangle")

    @classmethod
    def spacetime(cls, t_range: Tuple[float, float], *spatial_ranges, name: str = ""):
        """Spacetime domain with time as first coordinate."""
        bounds = [t_range] + list(spatial_ranges)
        return cls(bounds, name=name or "Spacetime")

    @classmethod
    def minisuperspace(cls, a_range: Tuple[float, float], *other_ranges, name: str = ""):
        """Minisuperspace for quantum cosmology."""
        bounds = [a_range] + list(other_ranges)
        return cls(bounds, name=name or "Minisuperspace")

    @classmethod
    def superspace(
        cls,
        fields: List[Tuple[float, float]],
        momenta: bool = False,
        metric: Optional[Callable] = None,
        field_names: Optional[List[str]] = None,
        name: str = "",
    ) -> "Domain":
        """Minisuperspace domain with metric structure for quantum cosmology.

        Creates a domain for multi-field Wheeler-DeWitt equations with
        proper superspace metric G^{AB} for the kinetic term.

        Args:
            fields: List of field ranges [(a_min, a_max), (φ_min, φ_max), ...]
                    First coordinate is typically the scale factor a
            momenta: If True, include conjugate momenta as coordinates
                    (phase space formulation for Moyal deformation)
            metric: Callable G(q) -> G^{AB} giving inverse metric tensor
                    Shape: (batch, dim, dim) symmetric matrix at each point
                    If None, uses flat metric δ^{AB}
            field_names: Optional names for fields, e.g., ["a", "φ", "χ"]

        Returns:
            Domain with superspace structure

        Example:
            # 2-field minisuperspace: scale factor a + inflaton φ
            domain = Domain.superspace(
                fields=[(0.1, 5.0), (-3.0, 3.0)],  # a ∈ [0.1, 5], φ ∈ [-3, 3]
                metric=lambda q: mx.stack([
                    mx.stack([-1/(24*q[:,0]**2), mx.zeros(q.shape[0])], axis=1),
                    mx.stack([mx.zeros(q.shape[0]), q[:,0]**3], axis=1)
                ], axis=1),  # G^{AB} for FLRW + scalar field
                field_names=["a", "φ"],
            )
        """
        bounds = list(fields)

        if momenta:
            # Phase space: double coordinates (q^A, p_A)
            # Add momentum ranges (typically unbounded, use large range)
            n_fields = len(fields)
            momentum_ranges = [(-10.0, 10.0)] * n_fields
            bounds = bounds + momentum_ranges

        domain = cls(bounds, name=name or "Superspace")

        # Store metric as attribute (not in dataclass fields for backward compat)
        domain._metric = metric
        domain._field_names = field_names or [f"q{i}" for i in range(len(fields))]
        domain._momenta = momenta
        domain._n_fields = len(fields)

        return domain

    def get_metric(self, q: mx.array) -> Optional[mx.array]:
        """Get the superspace metric at configuration points.

        Args:
            q: Configuration space points (batch, n_fields)

        Returns:
            Inverse metric G^{AB} of shape (batch, n_fields, n_fields)
            or None if flat metric
        """
        if hasattr(self, '_metric') and self._metric is not None:
            return self._metric(q)
        return None

    def get_metric_determinant(self, q: mx.array, h: float = 1e-4) -> mx.array:
        """Compute √|det G| for Laplacian-Beltrami factor ordering.

        Args:
            q: Configuration points (batch, n_fields)
            h: Step size for numerical determinant

        Returns:
            √|det G| of shape (batch,)
        """
        G = self.get_metric(q)
        if G is None:
            return mx.ones(q.shape[0])

        # For 1D, det = G^{00}
        if G.shape[1] == 1:
            return mx.sqrt(mx.abs(G[:, 0, 0]) + 1e-12)

        # For 2D, det = G^{00}G^{11} - (G^{01})²
        if G.shape[1] == 2:
            det = G[:, 0, 0] * G[:, 1, 1] - G[:, 0, 1] * G[:, 1, 0]
            return mx.sqrt(mx.abs(det) + 1e-12)

        # For higher dimensions, use recursive formula or approximation
        # Here we use a simple diagonal approximation for numerical stability
        det = mx.prod(mx.diagonal(G, axis1=1, axis2=2), axis=1)
        return mx.sqrt(mx.abs(det) + 1e-12)


@dataclass
class PDEProblem:
    """A PDE problem definition.

    Physics-first design: define the physics, not the numerics.

    Example:
        # Schrödinger equation
        problem = PDEProblem(
            name="Harmonic Oscillator",
            domain=Domain.interval(-5, 5),
            residual=lambda psi, x: -laplacian(psi, x)/2 + x**2 * psi(x)/2 - E * psi(x),
            boundary_conditions=[
                BoundaryCondition.dirichlet(lambda x: 0, where=lambda x: abs(x) > 4)
            ],
            parameters={"E": 0.5}  # Ground state energy
        )
    """
    name: str
    domain: Domain
    residual: Callable                          # PDE residual R[φ](x) = 0
    boundary_conditions: List[BoundaryCondition] = field(default_factory=list)
    parameters: Dict[str, float] = field(default_factory=dict)
    exact_solution: Optional[Callable] = None   # For validation
    description: str = ""

    def __post_init__(self):
        if not self.description:
            self.description = f"PDE: {self.name}"

    def compute_residual(self, model: Callable, x: mx.array) -> mx.array:
        """Compute PDE residual at points x."""
        return self.residual(model, x)

    def compute_bc_loss(self, model: Callable, x_boundary: mx.array, h: float = 1e-4) -> mx.array:
        """Compute boundary condition loss."""
        total_loss = mx.array(0.0)

        for bc in self.boundary_conditions:
            if bc.bc_type == BCType.DIRICHLET:
                # φ = value on boundary
                # Use soft masking since MLX doesn't support boolean indexing
                mask = bc.where(x_boundary).astype(mx.float32)

                # Compute predictions and targets for all points
                pred = model(x_boundary)
                target = bc.value(x_boundary)

                if len(pred.shape) > 1:
                    pred = pred[:, 0]
                if len(target.shape) > 1:
                    target = target[:, 0]

                # Weighted MSE with mask
                weighted_error = mask * (pred - target)**2
                n_boundary = mx.sum(mask) + 1e-8
                total_loss = total_loss + mx.sum(weighted_error) / n_boundary

            elif bc.bc_type == BCType.NEUMANN:
                # TODO: implement normal derivative
                pass

            elif bc.bc_type == BCType.PERIODIC:
                # φ(a) = φ(b)
                pass

            elif bc.bc_type == BCType.CAUCHY:
                # Initial conditions at t=0
                # Soft mask for t close to initial time
                mask = (x_boundary[:, 0] < self.domain.bounds[0][0] + 0.1).astype(mx.float32)

                pred = model(x_boundary)
                # For Cauchy, target is evaluated at spatial coords only
                # Simplified: evaluate at full coords
                target = bc.value(x_boundary)

                if len(pred.shape) > 1:
                    pred = pred[:, 0]
                if len(target.shape) > 1:
                    target = target[:, 0]

                weighted_error = mask * (pred - target)**2
                n_boundary = mx.sum(mask) + 1e-8
                total_loss = total_loss + mx.sum(weighted_error) / n_boundary

        return total_loss

    def to_latex(self) -> str:
        """Generate LaTeX description of the problem."""
        latex = f"\\textbf{{{self.name}}}\n\n"
        latex += f"Domain: ${self.domain.name}$\n\n"

        if self.boundary_conditions:
            latex += "Boundary conditions:\n"
            for bc in self.boundary_conditions:
                latex += f"  - {bc.name}\n"

        if self.parameters:
            latex += "Parameters:\n"
            for k, v in self.parameters.items():
                latex += f"  - ${k} = {v}$\n"

        return latex
