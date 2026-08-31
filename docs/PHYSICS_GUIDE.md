# Physics Guide

Comprehensive guide for using KAN-MLX-Physics to solve physics problems.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Physics-Informed Neural Networks (PINNs)](#physics-informed-neural-networks-pinns)
3. [The Batch-Grad Sum Trick](#the-batch-grad-sum-trick)
4. [PDE Solver DSL](#pde-solver-dsl)
5. [Quantum Mechanics](#quantum-mechanics)
6. [Quantum Cosmology](#quantum-cosmology)
7. [The Moyal Star Product](#the-moyal-star-product)
8. [Physics Symbolic Functions](#physics-symbolic-functions)
9. [Best Practices](#best-practices)
10. [Complete Examples](#complete-examples)

---

## Introduction

KAN-MLX-Physics was designed with physics applications in mind. Key features:

- **Pluggable basis functions** - Choose the optimal basis for your physics problem
- **Learnable basis functions** - Natural for representing wavefunctions
- **Symbolic regression** - Extract analytic formulas from trained networks
- **Efficient derivatives** - The batch-grad sum trick for PINNs
- **Physics DSL** - High-level interface for common PDEs
- **50+ physics functions** - Hermite, Bessel, spherical harmonics, etc.

### Choosing the Right Basis

| Physics Problem | Recommended Basis | Why |
|-----------------|-------------------|-----|
| Quantum harmonic oscillator | `hermite` with `weighted=True` | Eigenfunctions are Hermite × Gaussian |
| Hydrogen atom (radial) | `laguerre` with `alpha=2l+1` | Radial wavefunctions are Laguerre polynomials |
| Angular momentum | `legendre` | Spherical harmonics built from Legendre |
| Wave equations | `fourier` | Natural for periodic/oscillatory solutions |
| Spectral methods | `chebyshev` | Optimal for bounded domains |
| General PDEs | `bspline` (default) | Adaptive, works everywhere |

```python
from kan_mlx_physics.basis import recommend_basis

# Get automatic recommendations
bases = recommend_basis("quantum_mechanics", "radial")
# ['laguerre', 'hermite']

bases = recommend_basis("periodic")
# ['fourier']
```

---

## Physics-Informed Neural Networks (PINNs)

PINNs solve differential equations by training neural networks to satisfy:
1. The PDE residual at interior points
2. Boundary/initial conditions at the boundary
3. (Optional) Physical constraints (normalization, symmetry)

### PINNTrainer (Recommended)

The `PINNTrainer` class provides a high-level interface for PINN training:

```python
import mlx.core as mx
from kan_mlx_physics import MultKAN, PINNTrainer

# Create model
model = MultKAN(width=[1, 10, 10, 1], grid=5, k=3)

# Create trainer with finite differences (avoids nested autodiff issues)
trainer = PINNTrainer(
    model,
    lr=0.01,
    optimizer="adam",           # or "adamw", "sgd", "lbfgs"
    derivative_method="finite_diff",  # or "autodiff"
    compile=True,               # Use @mx.compile for speed
)

# Define PDE: u'' + u = 0 (harmonic oscillator)
def pde_loss(x):
    u = trainer.u(x)
    u_xx = trainer.d2u(x)
    pde_residual = u_xx + u

    # Boundary conditions: u(0) = 0, u(pi) = 0
    x_bc = mx.array([[0.0], [mx.pi]])
    u_bc = trainer.u(x_bc)
    bc_loss = mx.mean(u_bc**2)

    return mx.mean(pde_residual**2) + 100 * bc_loss

# Training loop
for step in range(1000):
    x = mx.random.uniform(low=0.0, high=mx.pi, shape=(500, 1))
    loss = trainer.step(pde_loss, x)

    if step % 100 == 0:
        print(f"Step {step}: loss = {loss:.6f}")

# Phase 2: Fine-tune with lower learning rate
trainer.lr = 0.001
for step in range(500):
    x = mx.random.uniform(low=0.0, high=mx.pi, shape=(500, 1))
    loss = trainer.step(pde_loss, x)
```

**Key PINNTrainer features:**
- **Dynamic learning rate**: Change `trainer.lr` between training phases
- **Multiple optimizers**: Adam, AdamW, SGD, or L-BFGS
- **Finite differences**: Use `derivative_method="finite_diff"` to avoid nested autodiff issues
- **Compilation**: Use `compile=True` for faster training
- **Auto-detect loss signature**: Loss functions don't need `model` as first argument

### L-BFGS for High Precision

For physics problems requiring high accuracy:

```python
trainer = PINNTrainer(model, optimizer="lbfgs", derivative_method="finite_diff")

# L-BFGS converges faster for physics problems
result = trainer.fit_lbfgs(pde_loss, x_train, max_iter=100)
print(f"Final loss: {result['loss']:.8f}")
```

### Basic PINN Setup (Low-Level API)

For more control, you can use the functional API directly:

```python
import mlx.core as mx
from kan_mlx_physics import MultKAN, get_params_list
from kan_mlx_physics.pinn import PINNOperators
from kan_mlx_physics.functional import functional_forward

# Create model
model = MultKAN(width=[1, 10, 10, 1], grid=5, k=3)
params = get_params_list(model)

# Create derivative operators
ops = PINNOperators(k=model.k, base_fun=model.base_fun, dim=1)

# Define PDE: u'' + u = 0 (harmonic oscillator)
def pde_residual(params, x):
    u = ops.u(params, x)
    u_xx = ops.d2x(params, x)
    return u_xx + u

# Boundary conditions: u(0) = 0, u(pi) = 0
def bc_loss(params):
    x_bc = mx.array([[0.0], [mx.pi]])
    u_bc = ops.u(params, x_bc)
    return mx.mean(u_bc**2)

# Total loss
def loss_fn(params, x_interior):
    pde = mx.mean(pde_residual(params, x_interior)**2)
    bc = bc_loss(params)
    return pde + 100 * bc
```

### Training Loop (Low-Level API)

```python
from kan_mlx_physics.functional import init_adam_state, adam_update

# Initialize
m, v, t = init_adam_state(params)
lr = 0.01

# Training
for step in range(1000):
    # Sample interior points
    x = mx.random.uniform(low=0.0, high=mx.pi, shape=(500, 1))

    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn)(params, x)

    # Update
    params, m, v, t = adam_update(params, grads, m, v, t, lr=lr)

    if step % 100 == 0:
        print(f"Step {step}: loss = {float(loss):.6f}")

# Update model
from kan_mlx_physics.functional import set_params_list
set_params_list(model, params)
```

---

## The Batch-Grad Sum Trick

The key innovation enabling efficient PINN training on MLX.

### The Problem

For PINNs, we need per-sample gradients:
```
du/dx for each sample in the batch
```

Naive approach (slow):
```python
grads = []
for xi in x_batch:
    grads.append(mx.grad(u)(params, xi))
# Very slow for large batches!
```

### The Solution

Instead of differentiating `u(x)` directly, differentiate the **sum**:

```python
def u_sum(params, x):
    return mx.sum(functional_forward(params, x, k, base_fun))

# Gradient w.r.t. x gives per-sample gradients
du_dx = mx.grad(u_sum, argnums=1)(params, x_batch)
```

### Why It Works

```
u_sum = u(x_0) + u(x_1) + u(x_2) + ...

d(u_sum)/dx = [du/dx_0, du/dx_1, du/dx_2, ...]
```

Each sample's gradient is independent - one backward pass gives all gradients!

### Second Derivatives

```python
def u_sum(params, x):
    return mx.sum(u(params, x))

def du_dx_sum(params, x):
    return mx.sum(mx.grad(u_sum, argnums=1)(params, x))

# Laplacian
d2u_dx2 = mx.grad(du_dx_sum, argnums=1)(params, x_batch)
```

### Performance

| Method | Time (1000 samples) |
|--------|---------------------|
| Loop | ~500ms |
| vmap (if available) | ~50ms |
| **Sum trick** | **~5ms** |

---

## PDEBuilder DSL (Recommended)

The most powerful way to solve physics problems. Provides a declarative, chainable API with trainable eigenvalues.

### Basic Usage

```python
from kan_mlx_physics.pde import PDEBuilder, PDEResidualLoss

model, history = (
    PDEBuilder("Derivative(u, x, 2) + k*u = 0")
    .params(k=1.0)
    .domain([0, 3.14159])
    .loss(PDEResidualLoss(weight=100))
    .phase("train", steps=1000, lr=0.01)
    .model(width=[1, 10, 1])
    .solve()
)
```

### Trainable Eigenvalues

The key feature for eigenvalue problems: train the eigenvalue E alongside the network weights.

```python
from kan_mlx_physics.pde import (
    PDEBuilder,
    PDEResidualLoss,
    BoundaryConditionLoss,
    NormalizationLoss,
    NonTrivialLoss,
    EigenvalueLoss,
)

L = 2.0  # Box length
E_analytic = np.pi**2 / (2 * L**2)  # ≈ 1.2337

model, history = (
    PDEBuilder("Derivative(psi, x, 2)/2 + E*psi = 0")
    .params(E=1.0)                    # Initial guess
    .trainable_params("E")            # Train E via Adam!
    .domain([0, L])
    .loss(PDEResidualLoss(weight=500, normalize=False))
    .loss(BoundaryConditionLoss(bc_type="dirichlet", weight=1000, target=0.0))
    .loss(NormalizationLoss(weight=100, target=1.0))
    .loss(NonTrivialLoss(weight=200))
    .loss(EigenvalueLoss(param_name="E", method="trainable", weight=3))
    .phase("initial", steps=2000, lr=0.003, n_points=1000, log_freq=400)
    .phase("refine", steps=1000, lr=0.0015, grid_update_before=True, log_freq=200)
    .model(width=[1, 2, 1], grid=5, k=3, grid_range=(0, L), seed=42)
    .solve(verbose=True)
)

# Access trained eigenvalue
E_trained = history.trainable_params["E"]
print(f"E = {E_trained:.6f}")  # E ≈ 1.2337 (0.00% error!)
print(f"Error: {abs(E_trained - E_analytic)/E_analytic*100:.2f}%")
```

### Available Loss Terms

| Loss | Purpose | Key Parameters |
|------|---------|----------------|
| `PDEResidualLoss` | PDE equation residual | `weight`, `normalize` |
| `BoundaryConditionLoss` | Dirichlet/Neumann BCs | `bc_type`, `target` |
| `NormalizationLoss` | ∫\|u\|² = target | `target` (default 1.0) |
| `NonTrivialLoss` | Prevent zero solution | `scale` (default 10.0) |
| `EigenvalueLoss` | Eigenvalue problems | `method`, `param_name` |
| `DecayLoss` | Decay at boundaries | `threshold_ratio` |
| `SmoothnessLoss` | Penalize oscillations | `weight` |
| `AnchorLoss` | Fix value at point | `x0`, `target` |

### Multi-Phase Training

```python
.phase("initial", steps=2000, lr=0.003, n_points=1000)
.phase("refine", steps=1000, lr=0.0015, grid_update_before=True)
.phase("final", steps=500, lr=0.0005)
```

### Equation Syntax

```python
# Standard derivative notation
"Derivative(u, x, 2) + u = 0"      # d²u/dx² + u = 0

# With parameters
"Derivative(psi, x, 2)/2 + E*psi = 0"  # Schrödinger

# Complex expressions (Wigner function)
"(r2 - 2*E)*W - (hbar**2/4)*(4*r2*Derivative(W, r2, 2) + 4*Derivative(W, r2)) = 0"
```

---

## Legacy PDE Solver (solve function)

For simpler use cases:

```python
from kan_mlx_physics.pde import solve

# Harmonic oscillator
psi, history = solve(
    "-nabla^2 psi/2 + x^2 psi/2 = E psi",
    domain=[-5, 5],
    params={"E": 0.5}
)

# Evaluate solution
x_test = mx.linspace(-5, 5, 100).reshape(-1, 1)
psi_vals = psi(x_test)
```

### Multi-Dimensional Domains

```python
# 2D Laplace equation
u, _ = solve(
    "laplace",
    domain={"x": [-1, 1], "y": [-1, 1]},
)

# Spacetime (wave equation)
u, _ = solve(
    "wave",
    domain={"t": [0, 2], "x": [-1, 1]},
    params={"c": 1.0},
)
```

---

## Quantum Mechanics

### Schrodinger Equation

Time-independent Schrodinger equation:
```
[-hbar^2/(2m) nabla^2 + V(x)] psi = E psi
```

**Recommended Basis:** Use `hermite` with `weighted=True` for harmonic oscillator problems, as the exact solutions are Hermite polynomials times Gaussians.

```python
from kan_mlx_physics import MultKAN
from kan_mlx_physics.pde import solve

# Harmonic oscillator with physics-optimized basis
model = MultKAN(
    width=[1, 20, 20, 1],
    basis="hermite",
    basis_M=10,
    basis_kwargs={"weighted": True}  # Multiply by exp(-x²/2)
)

# Harmonic oscillator ground state
psi, history = solve(
    "-nabla^2 psi/2 + x^2 psi/2 = E psi",
    domain=[-5, 5],
    params={"E": 0.5},  # Ground state energy
    model=model,        # Use our Hermite-based model
)

# Verify energy eigenvalue
x = mx.linspace(-5, 5, 100).reshape(-1, 1)
psi_vals = psi(x)

# Should be Gaussian: psi_0 ~ exp(-x^2/2)
import matplotlib.pyplot as plt
plt.plot(x, psi_vals, label='KAN')
plt.plot(x, mx.exp(-x**2/2), '--', label='Exact')
plt.legend()
plt.show()
```

### Particle in a Box

```python
# Particle in infinite square well [0, L]
L = 1.0
psi, history = solve(
    "-nabla^2 psi/2 = E psi",
    domain=[0, L],
    params={"E": mx.pi**2 / 2}  # Ground state
)

# Boundary conditions automatically: psi(0) = psi(L) = 0
# Solution: psi_n(x) = sqrt(2/L) * sin(n*pi*x/L)
```

### Eigenvalue Problems

Find multiple eigenvalues:

```python
from kan_mlx_physics.pde import solve_eigenvalue

models, eigenvalues, histories = solve_eigenvalue(
    "harmonic",
    domain=[-5, 5],
    n_eigenvalues=5,
)

# E_n = (n + 1/2) * hbar * omega
for n, E in enumerate(eigenvalues):
    print(f"E_{n} = {E:.4f} (exact: {n + 0.5:.4f})")
```

---

## Quantum Cosmology

### Wheeler-DeWitt Equation

The "Schrodinger equation of the universe":
```
H Psi = 0
```

In minisuperspace (single scale factor a):
```
[-hbar^2 d^2/da^2 + U(a)] Psi(a) = 0
```

```python
from kan_mlx_physics.pde import solve

# Wheeler-DeWitt with cosmological constant
Psi, history = solve(
    "wheeler-dewitt",
    domain=[0.1, 10],  # Scale factor range
    params={
        "Lambda": 0.01,  # Cosmological constant
        "k": 1,          # Spatial curvature
    }
)
```

### Deformed Wheeler-DeWitt

With Moyal star product for deformation quantization:
```
H *_theta Psi = 0
```

```python
# Deformed Wheeler-DeWitt
Psi, history = solve(
    "H star_theta Psi = 0",
    domain=[0.1, 5],
    params={"theta": 0.05}  # Deformation parameter
)
```

### Friedmann Equations

Classical cosmology:
```
H^2 = 8*pi*G*rho/3 - k/a^2 + Lambda/3
```

```python
from kan_mlx_physics.pde.physics import Friedmann

problem = Friedmann(
    Lambda=0.7,           # Dark energy
    Omega_m=0.3,          # Matter density
    k=0,                  # Flat universe
)

a, history = solve(problem, domain=[0.01, 10])
```

---

## The Moyal Star Product

For deformation quantization, the Moyal star product replaces ordinary multiplication:

### Definition

```
f *_theta g = f * g + (i*theta/2) {f, g}_P + O(theta^2)
```

where `{f, g}_P` is the Poisson bracket.

### Implementation

```python
from kan_mlx_physics.pde.operators import star_product, moyal_bracket

# Define functions in phase space (q, p)
def f(x):
    return x[:, 0]**2  # q^2

def g(x):
    return x[:, 1]**2  # p^2

# Phase space points
x = mx.random.uniform(shape=(100, 2))

# Moyal bracket {f, g}_M
bracket = moyal_bracket(f, g, x, hbar=0.1)

# Star product f *_theta g
product = star_product(f, g, x, hbar=0.1, order=2)
```

### Orders of Expansion

| Order | Includes |
|-------|----------|
| 0 | Classical product f*g |
| 1 | + (theta/2) Poisson bracket |
| 2 | + O(theta^2) corrections |

### Using in PDEs

```python
from kan_mlx_physics.pde.physics import DeformedWheelerDeWitt

# Wheeler-DeWitt with Moyal deformation
problem = DeformedWheelerDeWitt(
    theta=0.05,  # Deformation parameter
    hbar=1.0,
)

Psi, history = solve(problem, domain=[(0.1, 5), (-5, 5)])
```

---

## Physics Symbolic Functions

50+ physics-relevant functions for symbolic regression.

### Registration

```python
from kan_mlx_physics.physics_symbolic import register_physics_symbolic, list_physics_symbolic

# Register all physics functions (call once)
register_physics_symbolic()

# See available functions
physics_fns = list_physics_symbolic()
for category, fns in physics_fns.items():
    print(f"{category}: {fns}")
```

### Categories

**Quantum Mechanics:**
- Hermite polynomials: `H_0`, `H_1`, `H_2`, `H_3`, `H_4`
- Harmonic oscillator wavefunctions: `psi_0`, `psi_1`, `psi_2`
- Laguerre polynomials: `L_0`, `L_1`, `L_2`
- Hydrogen radial: `R_10`, `R_20`, `R_21`
- Legendre: `P_0`, `P_1`, `P_2`, `P_3`, `P_4`
- Chebyshev: `T_0`, `T_1`, `T_2`, `T_3`

**Special Functions:**
- Bessel: `J_0`, `J_1`, `Y_0`, `Y_1`, `I_0`, `I_1`, `K_0`, `K_1`
- Spherical Bessel: `j_0`, `j_1`
- Airy: `Ai`, `Bi`
- Error functions: `erf`, `erfc`
- Gamma: `gamma`, `loggamma`, `digamma`
- Elliptic: `ellipK`, `ellipE`

**QFT:**
- Propagators: `propagator`, `yukawa`, `coulomb`
- Statistics: `bose`, `fermi`, `planck`
- Polylogarithm: `Li_2`

**Cosmology:**
- Scale factor: `a_matter`, `a_rad`, `a_deSitter`
- Distance: `D_L`
- Metric: `schwarzschild`

**Deformation:**
- q-exponential: `q_exp_0`, `q_exp_2`
- q-logarithm: `q_log_2`
- q-trigonometric: `sin_q`, `cos_q`
- Moyal basis: `moyal_1`, `moyal_2`

### Using in Symbolic Regression

```python
# Register physics functions
register_physics_symbolic()

# Train model
model.fit(dataset, steps=1000)

# Now auto_symbolic can find physics functions
model.auto_symbolic(x_sample, r2_threshold=0.9)

# If the learned function is psi_0(x) = exp(-x^2/2):
formula = model.symbolic_formula(['x'])
# Could output: psi_0(1.00*x)
```

---

## Best Practices

### 1. Choosing Architecture

| Problem Type | Recommended Width |
|--------------|-------------------|
| 1D ODE | `[1, 10, 1]` |
| 1D PDE | `[1, 20, 20, 1]` |
| 2D PDE | `[2, 32, 32, 1]` |
| Eigenvalue | `[1, 20, 20, 1]` |
| High-frequency | `[1, 32, 32, 32, 1]` |

### 2. Training Strategy

```python
# Phase 1: Coarse training
history1 = model.fit(dataset, steps=500, lr=0.01, grid=5)

# Phase 2: Grid refinement
model.refine(new_grid=10)
history2 = model.fit(dataset, steps=500, lr=0.005)

# Phase 3: Symbolic discovery
model.auto_symbolic(x_sample)

# Phase 4: Fine-tuning with fixed symbols
history3 = model.fit(dataset, steps=200, lr=0.001)
```

### 3. Loss Weighting

```python
# PDE residual should be well-balanced with BC loss
# Start with:
lambda_pde = 1.0
lambda_bc = 10.0  # BCs are hard constraints

# If BC not satisfied, increase lambda_bc
# If PDE residual dominates, decrease lambda_pde
```

### 4. Domain Normalization

```python
# Always normalize domain to [-1, 1] or [0, 1]
# Original domain: [0, L] with L = 10

def normalize(x):
    return 2 * x / L - 1  # Maps [0, L] to [-1, 1]

def denormalize(x_norm):
    return (x_norm + 1) * L / 2

# Train on normalized domain
x_train_norm = normalize(x_train)
```

### 5. Numerical Stability

```python
# Use singularity-avoiding forward pass near problematic regions
y = model(x, singularity_avoiding=True, y_th=10.0)

# Use safe symbolic functions
from kan_mlx_physics.symbolic import SYMBOLIC_REGISTRY
sqrt_fn = SYMBOLIC_REGISTRY['x^0.5']
sqrt_safe = sqrt_fn.safe_fn  # Handles x < 0
```

---

## Complete Examples

### Example 1: Quantum Harmonic Oscillator

```python
import mlx.core as mx
from kan_mlx_physics import MultKAN, get_params_list, set_params_list
from kan_mlx_physics.pinn import PINNOperators
from kan_mlx_physics.functional import init_adam_state, adam_update

# Parameters
L = 5.0  # Domain: [-L, L]
E = 0.5  # Ground state energy

# Model
model = MultKAN(width=[1, 20, 20, 1], grid=5, k=3)
params = get_params_list(model)
ops = PINNOperators(k=model.k, base_fun=model.base_fun, dim=1)

# Loss function
def loss_fn(params, x):
    u = ops.u(params, x)
    u_xx = ops.d2x(params, x)

    # Schrodinger: -u''/2 + x^2*u/2 = E*u
    V = 0.5 * x[:, 0]**2
    residual = -0.5 * u_xx + V * u - E * u

    # Normalization constraint
    norm = mx.sum(u**2) * (2*L / len(u))

    return mx.mean(residual**2) + 10 * (norm - 1)**2

# Training
m, v, t = init_adam_state(params)
for step in range(2000):
    x = mx.random.uniform(low=-L, high=L, shape=(500, 1))
    loss, grads = mx.value_and_grad(loss_fn)(params, x)
    params, m, v, t = adam_update(params, grads, m, v, t, lr=0.01)

    if step % 200 == 0:
        print(f"Step {step}: loss = {float(loss):.6f}")

set_params_list(model, params)

# Symbolic regression
from kan_mlx_physics.physics_symbolic import register_physics_symbolic
register_physics_symbolic()
model.auto_symbolic(x)
print(model.symbolic_formula(['x']))
# Expected: psi_0(x) = exp(-x^2/2)
```

### Example 2: Particle in a Box

```python
import mlx.core as mx
from kan_mlx_physics import MultKAN, get_params_list, set_params_list
from kan_mlx_physics.pinn import make_derivative_fns
from kan_mlx_physics.functional import functional_forward, init_adam_state, adam_update

L = 1.0  # Box length
n = 1    # Quantum number
E_exact = (n * mx.pi / L)**2 / 2

model = MultKAN(width=[1, 16, 1], grid=5, k=3)
params = get_params_list(model)
u_fn, du_dx, d2u_dx2 = make_derivative_fns(model.k, model.base_fun)

# Eigenvalue as trainable parameter
E = mx.array([1.0])

def loss_fn(params, E, x):
    u = u_fn(params, x)
    u_xx = d2u_dx2(params, x)

    # PDE: -u''/2 = E*u
    pde = mx.mean((-0.5 * u_xx - E * u)**2)

    # Boundary conditions: u(0) = u(L) = 0
    x_bc = mx.array([[0.0], [L]])
    u_bc = u_fn(params, x_bc)
    bc = mx.sum(u_bc**2)

    # Normalization
    norm = mx.mean(u**2)
    norm_loss = (norm - 1)**2

    return pde + 100 * bc + 10 * norm_loss

# Train
m, v, t = init_adam_state(params)
lr = 0.01

for step in range(2000):
    x = mx.random.uniform(low=0.01, high=L-0.01, shape=(300, 1))

    def total_loss(params):
        return loss_fn(params, E, x)

    loss, grads = mx.value_and_grad(total_loss)(params)
    params, m, v, t = adam_update(params, grads, m, v, t, lr=lr)

    # Also update E
    E_grad = mx.grad(lambda e: loss_fn(params, e, x))(E)
    E = E - 0.001 * E_grad

    if step % 200 == 0:
        print(f"Step {step}: loss = {float(loss):.4f}, E = {float(E):.4f}")

print(f"\nLearned E = {float(E):.4f}")
print(f"Exact E = {float(E_exact):.4f}")
```

### Example 3: Wheeler-DeWitt with Moyal Deformation

```python
import mlx.core as mx
from kan_mlx_physics import MultKAN, get_params_list, set_params_list
from kan_mlx_physics.pde.operators import star_product
from kan_mlx_physics.functional import functional_forward, init_adam_state, adam_update

# Minisuperspace model: 2D phase space (a, pi_a)
theta = 0.05  # Deformation parameter

model = MultKAN(width=[2, 20, 20, 1], grid=5, k=3)
params = get_params_list(model)

def psi_fn(params, x):
    return functional_forward(params, x, model.k, model.base_fun)

# Hamiltonian: H = pi^2/2 - U(a)
# U(a) = a (1 - Lambda * a^2) for de Sitter
Lambda = 0.1

def H_classical(x):
    a = x[:, 0:1]
    pi = x[:, 1:2]
    return 0.5 * pi**2 - a * (1 - Lambda * a**2)

def loss_fn(params, x):
    # H *_theta Psi = 0
    def psi(y):
        return psi_fn(params, y)

    # Compute star product (to second order)
    H_star_psi = star_product(H_classical, psi, x, hbar=theta, order=2)

    return mx.mean(H_star_psi**2)

# Training
m, v, t = init_adam_state(params)

for step in range(3000):
    # Sample phase space
    a = mx.random.uniform(low=0.1, high=3.0, shape=(200, 1))
    pi = mx.random.uniform(low=-2.0, high=2.0, shape=(200, 1))
    x = mx.concatenate([a, pi], axis=1)

    loss, grads = mx.value_and_grad(loss_fn)(params, x)
    params, m, v, t = adam_update(params, grads, m, v, t, lr=0.005)

    if step % 300 == 0:
        print(f"Step {step}: loss = {float(loss):.6f}")

set_params_list(model, params)
print("Deformed Wheeler-DeWitt solved!")
```

---

## Further Reading

- **KAN Paper:** [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- **PINNs:** Raissi et al., "Physics-informed neural networks"
- **Wheeler-DeWitt:** DeWitt, "Quantum Theory of Gravity"
- **Moyal Product:** Groenewold-Moyal formalism in phase space QM
- **MLX Framework:** [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
