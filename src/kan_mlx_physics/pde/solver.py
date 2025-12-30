"""Physics-informed KAN solver for PDEs.

Elegant interface: define physics, get solutions.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Optional, Dict, Callable, Tuple, List
from dataclasses import dataclass

from .problem import PDEProblem, Domain


@dataclass
class SolverConfig:
    """Solver configuration for physics-informed KAN.

    The default values are tuned for typical 1D quantum mechanics problems.
    For higher-dimensional or more complex PDEs, consider increasing network
    width and training steps.
    """
    # Network architecture
    width: List[int] = None       # KAN architecture (default: [1, 10, 10, 1])
    grid: int = 10                # Spline grid points per layer
    k: int = 3                    # Spline order (3=cubic, good balance of smoothness/flexibility)

    # Training hyperparameters
    steps: int = 1000             # Training iterations
    lr: float = 0.01              # Learning rate (0.01 works well for Adam on most PDEs)
    optimizer: str = "Adam"       # Optimizer: "Adam" or "SGD"

    # Collocation point sampling
    n_interior: int = 1000        # Interior collocation points (more = better PDE fit)
    n_boundary: int = 200         # Boundary collocation points
    sampling: str = "sobol"       # Sampling method: "uniform", "sobol" (quasi-random), "adaptive"

    # Loss function weights (critical for convergence)
    lambda_pde: float = 1.0       # PDE residual weight (baseline)
    lambda_bc: float = 10.0       # Boundary condition weight (10x higher to enforce BCs strongly,
                                  # since BCs are exact constraints while PDE is approximate)
    lambda_reg: float = 0.001     # L2 regularization weight (prevents overfitting to noise)

    # Complex wavefunction support (for Schrödinger, Wheeler-DeWitt)
    complex_output: bool = False  # If True, model outputs [Re(Ψ), Im(Ψ)]

    # Moyal star product for quantum cosmology
    star_product_order: int = 2   # Expansion order: 2=semiclassical, 4/6=higher corrections

    # Logging
    log_freq: int = 100           # Print loss every N steps
    verbose: bool = True          # Enable progress output

    def __post_init__(self):
        """Set default width if not specified."""
        if self.width is None:
            # Default: 1 input -> 10 hidden -> 10 hidden -> 1 output
            # Good for 1D problems; increase for higher dimensions
            self.width = [1, 10, 10, 1]


def solve(
    problem: PDEProblem,
    config: Optional[SolverConfig] = None,
    model: Optional["MultKAN"] = None,
    seed: int = 42
) -> Tuple["MultKAN", Dict]:
    """Solve a PDE using physics-informed KAN.

    Args:
        problem: PDEProblem to solve
        config: Solver configuration
        model: Pre-existing model (optional)
        seed: Random seed

    Returns:
        (trained_model, history)

    Example:
        from kan_mlx_physics.pde import WheelerDeWitt, solve

        problem = WheelerDeWitt()
        model, history = solve(problem)

        # Evaluate solution
        a = mx.linspace(0.1, 5, 100).reshape(-1, 1)
        psi = model(a)

        # Complex wavefunction example:
        from kan_mlx_physics.pde.complex import ComplexWavefunction

        config = SolverConfig(complex_output=True)
        model, _ = solve(problem, config)
        psi = ComplexWavefunction(model)
        amp = psi.amplitude(a)
        phase = psi.phase(a)
    """
    from ..multkan import MultKAN
    from .complex import complex_residual

    if config is None:
        config = SolverConfig()

    # Adjust network input/output dims to problem
    dim = problem.domain.dim
    width = config.width.copy()
    width[0] = dim

    # Output dimension: 2 for complex, 1 for real
    if config.complex_output:
        width[-1] = 2
    else:
        width[-1] = 1

    # Create model
    if model is None:
        model = MultKAN(
            width=width,
            grid=config.grid,
            k=config.k,
            seed=seed
        )

    # Sample collocation points
    x_interior = problem.domain.sample(config.n_interior, config.sampling, seed)
    x_boundary = problem.domain.sample_boundary(config.n_boundary, seed)

    # History
    history = {
        "loss": [],
        "pde_loss": [],
        "bc_loss": [],
        "step": []
    }

    # Optimizer
    if config.optimizer == "Adam":
        optimizer = optim.Adam(learning_rate=config.lr)
    elif config.optimizer == "AdamW":
        optimizer = optim.AdamW(learning_rate=config.lr)
    else:
        optimizer = optim.SGD(learning_rate=config.lr)

    # Loss function for gradient computation
    def loss_fn(model, x_int, x_bnd):
        # PDE residual loss
        if config.complex_output:
            # For complex wavefunctions, compute residual for both components
            residual = complex_residual(
                lambda m, x: problem.compute_residual(m, x),
                model, x_int
            )
        else:
            residual = problem.compute_residual(model, x_int)
            if len(residual.shape) > 1:
                residual = residual[:, 0]
        pde_loss = mx.mean(residual**2)

        # Boundary condition loss
        if config.complex_output:
            # For complex, enforce BC on both components
            out_bnd = model(x_bnd)
            psi_R, psi_I = out_bnd[:, 0], out_bnd[:, 1]
            # Default: Dirichlet Ψ = 0 at boundaries
            bc_loss = mx.mean(psi_R**2 + psi_I**2)
        else:
            bc_loss = problem.compute_bc_loss(model, x_bnd)

        # Regularization (sparsity)
        reg_loss = mx.array(0.0)
        for layer in model.layers:
            reg_loss = reg_loss + mx.mean(mx.abs(layer.coef))

        total = (config.lambda_pde * pde_loss +
                 config.lambda_bc * bc_loss +
                 config.lambda_reg * reg_loss)

        return total

    # Wrapper for value_and_grad
    def loss_wrapper(model, x_int, x_bnd):
        return loss_fn(model, x_int, x_bnd)

    # Separate function to compute detailed losses for logging
    def compute_losses(model, x_int, x_bnd):
        residual = problem.compute_residual(model, x_int)
        if len(residual.shape) > 1:
            residual = residual[:, 0]
        pde_loss = mx.mean(residual**2)
        bc_loss = problem.compute_bc_loss(model, x_bnd)
        return pde_loss, bc_loss

    # Training loop
    if config.verbose:
        print(f"\nSolving: {problem.name}")
        print(f"Domain: {problem.domain.name}")
        print(f"Network: {width}")
        print("-" * 50)

    loss_and_grad_fn = nn.value_and_grad(model=model, fn=loss_wrapper)

    for step in range(config.steps):
        # Compute loss and gradients
        loss, grads = loss_and_grad_fn(model, x_interior, x_boundary)

        # Update
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        # Resample periodically for better coverage
        if step > 0 and step % 200 == 0:
            x_interior = problem.domain.sample(config.n_interior, config.sampling, seed + step)

        # Log
        if step % config.log_freq == 0:
            pde_loss, bc_loss = compute_losses(model, x_interior, x_boundary)
            mx.eval(pde_loss, bc_loss)

            history["loss"].append(float(loss))
            history["pde_loss"].append(float(pde_loss))
            history["bc_loss"].append(float(bc_loss))
            history["step"].append(step)

            if config.verbose:
                print(f"Step {step:4d} | Loss: {float(loss):.2e} | "
                      f"PDE: {float(pde_loss):.2e} | BC: {float(bc_loss):.2e}")

    if config.verbose:
        print("-" * 50)
        print(f"Final loss: {float(loss):.2e}")

    return model, history


def solve_eigenvalue(
    problem: PDEProblem,
    n_eigenvalues: int = 5,
    config: Optional[SolverConfig] = None,
    seed: int = 42
) -> Tuple[List["MultKAN"], List[float], Dict]:
    """Solve eigenvalue PDE (like Schrödinger).

    Uses iterative approach to find multiple eigenvalues.

    Args:
        problem: Eigenvalue PDE problem
        n_eigenvalues: Number of eigenvalues to find
        config: Solver configuration
        seed: Random seed

    Returns:
        (models, eigenvalues, history)
    """
    from ..multkan import MultKAN

    if config is None:
        config = SolverConfig()
        config.steps = 500  # Shorter for each eigenvalue

    models = []
    eigenvalues = []
    histories = []

    # Initial eigenvalue guess
    E = 0.5

    for n in range(n_eigenvalues):
        if config.verbose:
            print(f"\n{'='*50}")
            print(f"Finding eigenvalue {n}")
            print(f"{'='*50}")

        # Update problem with current energy guess
        problem.parameters["E"] = E

        # Solve
        model, history = solve(problem, config, seed=seed + n)

        # Estimate eigenvalue from solution
        # (Simple approach: use variational principle)
        x_test = problem.domain.sample(500, "sobol", seed)
        psi = model(x_test)
        if len(psi.shape) > 1:
            psi = psi[:, 0]

        # Normalize
        psi_norm = psi / (mx.sqrt(mx.mean(psi**2)) + 1e-10)

        models.append(model)
        eigenvalues.append(E)
        histories.append(history)

        # Next eigenvalue guess (simple increment)
        E = E + 1.0

    return models, eigenvalues, histories


def adaptive_solve(
    problem: PDEProblem,
    config: Optional[SolverConfig] = None,
    tol: float = 1e-4,
    max_refinements: int = 3,
    seed: int = 42
) -> Tuple["MultKAN", Dict]:
    """Solve with adaptive refinement.

    Refines collocation points in regions of high residual.

    Args:
        problem: PDE problem
        config: Solver configuration
        tol: Residual tolerance
        max_refinements: Maximum refinement iterations
        seed: Random seed

    Returns:
        (model, history)
    """
    if config is None:
        config = SolverConfig()

    model = None
    history = {"loss": [], "refinements": []}

    for ref in range(max_refinements + 1):
        if config.verbose:
            print(f"\n--- Refinement {ref} ---")

        model, hist = solve(problem, config, model, seed + ref)

        # Check residual
        x_test = problem.domain.sample(1000, "sobol", seed)
        residual = problem.compute_residual(model, x_test)
        max_res = float(mx.max(mx.abs(residual)))

        history["loss"].extend(hist["loss"])
        history["refinements"].append({"step": ref, "max_residual": max_res})

        if config.verbose:
            print(f"Max residual: {max_res:.2e}")

        if max_res < tol:
            if config.verbose:
                print(f"Converged at refinement {ref}")
            break

        # Adaptive sampling: increase points in high-residual regions
        config.n_interior = int(config.n_interior * 1.5)

    return model, history
