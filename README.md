# KAN-MLX-Physics

**A fast, MLX-native Kolmogorov–Arnold PINN toolkit for Apple Silicon.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.21+-orange.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

KAN-MLX-Physics is a research-oriented implementation of [Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756) (KANs) for physics-informed learning, eigenvalue problems, and scientific discovery. It combines an MLX-native KAN implementation with a declarative physics/PDE interface designed around the kinds of small-to-medium scientific models for which KANs are especially interesting.

### Why this exists

The KAN ecosystem already has excellent tools, but they target somewhat different use cases. [PyKAN](https://github.com/KindXiaoming/pykan) is the reference **PyTorch** implementation of KANs and provides a rich environment for KAN research and symbolic analysis. [KINN/PIKAN](https://arxiv.org/abs/2406.11045) established KANs as physics-informed approximators for forward and inverse PDE problems, framed as a *method* built on existing KAN implementations rather than as a standalone MLX package.

KAN-MLX-Physics occupies a narrower niche:

> **KAN-based scientific machine learning with an Apple-Silicon-first MLX implementation and a physics-oriented API.**

The intellectual identity is not "runs on Apple Silicon" — that is the *engineering niche* — but the physics workflow it operationalizes: **basis priors, grow/prune architecture discovery, trainable-eigenvalue discovery, Hilbert–Schmidt deflation, Wigner/Moyal operators, and symbolic recovery.** On Apple Silicon, MLX's shared-memory execution model also lets CPU and GPU operations work on the same arrays without explicit host↔device transfers, which is attractive for workloads built from small networks and frequently-resampled collocation sets.

> **Note:** An independent implementation inspired by [PyKAN](https://github.com/KindXiaoming/pykan); not affiliated with the `mlx-kan` package on PyPI.

### Scope & limitations

This is research software with an intentionally narrow scope.

- **Apple Silicon is the primary supported and tested target.** The library is developed around MLX on M-series Macs. MLX itself now exposes additional backends (Linux CPU, CUDA), but portability of *this package* to them is not currently a reproducibility guarantee.
- **Higher-order autodiff is expensive.** Physics-informed objectives with second derivatives require nested automatic differentiation; in the current implementation these terms can dominate training cost and substantially reduce steps/second versus first-order objectives.
- **Small scientific models are the target** — KAN/PINN research, eigenvalue problems, interpretable models, symbolic recovery — not large-scale deep-learning training.
- **Research-grade rather than production-grade.** Core KAN and physics paths are tested and actively used; pruning, broader backend validation, and some docs remain under development.
- **Reproducibility is hardware-sensitive.** Performance should always be reported with the Apple chip, MLX version, model architecture, collocation count, derivative order, and precision.

If those trade-offs match your problem — particularly KAN-based physics-informed learning on Apple hardware — this library is built specifically for that use case.

### Performance

Initial microbenchmarks on an **Apple M3 Max** illustrate the main computational distinction between ordinary KAN optimization and physics-informed objectives requiring higher-order derivatives.

| Workload (`[1,2,1]`, 2500 collocation points) | Approx. time / step | Notes |
|---|---:|---|
| PDE-residual term only | ≈ 11 ms | first-order-dominated |
| Full physics-informed step (PDE + trace + purity + BC + …) | ≈ 18 ms | complete training step |
| Second-derivative overhead (isolated `W''` loss vs `W'`) | ≈ 2× | *not* an order-of-magnitude penalty |

These numbers are **not cross-framework benchmarks** and should not be read as evidence that KAN-MLX-Physics is faster than PyKAN, PyTorch, or JAX. They are un-contended per-step timings on one M3 Max; run several trainings concurrently and per-step time rises with GPU contention. For reproducible comparisons, see the scripts and environment notes in `benchmarks/`.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Apple Silicon Optimized** | Native Metal acceleration via MLX, reduced memory overhead with unified memory |
| **Pluggable Basis Functions** | B-splines, Fourier, Chebyshev, Hermite, Laguerre, Legendre - choose the right basis for your physics |
| **Physics-First Design** | PDE DSL with trainable eigenvalues, 50+ physics symbolic functions, Moyal star product |
| **Efficient Derivatives** | Batch-grad sum trick for PINN training |
| **PyKAN Compatible** | Similar API patterns, identical visualization style |
| **Functional API** | Pure functions for `mx.grad()`, stateless parameter handling |
| **Multi-Format Output** | Formula export to Unicode, LaTeX, Typst, SymPy |

---

## Quick Install

```bash
# From source (recommended)
git clone git@github.com:YaelMtz/kan-mlx-physics.git
cd kan-mlx-physics
pip install -e "."

# With optional dependencies
pip install -e ".[dev]"      # Development tools
pip install -e ".[viz]"      # Visualization (matplotlib)
pip install -e ".[physics]"  # Physics extras (sympy)
pip install -e ".[all]"      # Everything
```

**Requirements:** macOS with Apple Silicon (M1/M2/M3/M4), Python 3.10+, MLX 0.21+

**Import Options:**
```python
# Standard import
from kan_mlx_physics import MultKAN, PINNTrainer, create_dataset

# Shorthand alias (recommended for interactive use)
import kan_mlx_physics as kmp
from kan_mlx_physics import MultKAN, PINNTrainer, quick_fit
```

---

## 30-Second Example

```python
from kan_mlx_physics import quick_fit, auto_formula

# One-liner: create, train, and get model
model, history = quick_fit(
    f=lambda x: mx.sin(mx.pi * x[:, 0]) + x[:, 1]**2,
    n_var=2,
)

# Extract symbolic formula
formula = auto_formula(model, history['x_sample'], ['x', 'y'])
print(formula)  # sin(3.14*x) + y^2
```

### Verbose Example (Full Control)

```python
import mlx.core as mx
from kan_mlx_physics import MultKAN, create_dataset

# Create synthetic dataset: f(x,y) = sin(pi*x) + y^2
dataset = create_dataset(
    f=lambda x: mx.sin(mx.pi * x[:, 0]) + x[:, 1]**2,
    n_var=2,
    train_num=1000
)

# Build and train KAN (default B-spline basis)
model = MultKAN(width=[2, 5, 1], grid_points=5, spline_order=3)
history = model.fit(dataset, steps=500, learning_rate=0.01)

# Extract symbolic formula
model.auto_symbolic(dataset['train_input'])
print(model.symbolic_formula(['x', 'y']))
# Output: sin(3.14*x) + y^2
```

### Physics Presets

```python
from kan_mlx_physics import from_preset, list_presets

# See available presets
print(list_presets())
# {'quantum_oscillator': 'Hermite basis for QM',
#  'wave_equation': 'Fourier basis for waves',
#  'spectral': 'Chebyshev for high-accuracy', ...}

# Create model from preset
model = from_preset("quantum_oscillator", width=[1, 10, 1])
model = from_preset("wave_equation", width=[2, 15, 1])
```

### Physics-Optimized Basis Functions

```python
# FourierKAN for periodic/oscillatory functions (wave equations)
model = MultKAN(width=[2, 10, 1], basis="fourier", basis_M=11)

# HermiteKAN for quantum harmonic oscillator
model = MultKAN(width=[1, 10, 1], basis="hermite", basis_kwargs={"weighted": True})

# ChebyshevKAN for spectral methods
model = MultKAN(width=[2, 10, 1], basis="chebyshev", basis_M=8)

# LaguerreKAN for radial problems (hydrogen atom)
model = MultKAN(width=[1, 10, 1], basis="laguerre", basis_kwargs={"alpha": 1.0})

# Per-layer basis: Chebyshev input, Fourier hidden
model = MultKAN(
    width=[2, 10, 10, 1],
    basis=["chebyshev", "fourier", "bspline"],
    basis_M=[8, 15, 10]
)
```

---

## Performance

Per-step timings on an Apple M3 Max are given in the
[Performance table](#performance-apple-m3-max-mlx) above. The headline points:

- **Unified memory removes the host↔device copy** that dominates small-batch PINN
  workloads on discrete-GPU stacks — the main structural advantage of MLX here.
- **Second-order-autodiff carries a real but modest overhead** (≈2× over an
  otherwise-identical first-order loss, measured in isolation — *not* an
  order-of-magnitude ceiling; see the ≈11 ms vs ≈18 ms figures above). MLX cannot
  yet `compile` a step containing a nested `vjp`, so the physics loop runs eagerly;
  this is a framework limit, not a hardware one.

### Note on comparing to PyKAN

A meaningful head-to-head is *hardware-dependent* and easy to get wrong.
[PyKAN](https://github.com/KindXiaoming/pykan) has no working Apple-Metal path —
`torch.linalg.lstsq` is unimplemented on MPS (see below), so it falls back to CPU
on a Mac. Comparing MLX-on-Metal to PyKAN-on-CPU therefore reflects *backend
availability* on Apple hardware, not algorithmic superiority. In an
apples-to-apples steady-state comparison the picture is a **crossover**: MLX tends
to pull ahead on larger models and batches, while PyKAN on CPU can win for very
small networks. We do not quote a single speedup number; run
[`benchmarks/compare_pykan.py`](benchmarks/compare_pykan.py) on your own hardware
and report what you see.

> **Symbolic regression** is not timing-comparable: PyKAN uses exhaustive library
> search, this package uses R² fitting against a physics-motivated vocabulary.

#### PyKAN MPS Compatibility Issues

PyKAN has known compatibility issues with Apple's MPS (Metal Performance Shaders) backend:

- **`lstsq` not supported**: PyTorch's `torch.linalg.lstsq()` is not implemented for MPS, causing training and symbolic regression to fail
- **Workaround**: PyKAN benchmarks run on CPU, while KAN-MLX-Physics uses native Metal acceleration
- **Impact**: This gives KAN-MLX-Physics a significant advantage on Apple Silicon, as it can fully utilize the GPU

If you encounter `RuntimeError: lstsq failed` or similar errors with PyKAN on macOS, set:
```python
device = torch.device("cpu")  # Force CPU for PyKAN
```

#### Why is KAN-MLX-Physics Faster?

1. **Unified Memory**: No CPU-GPU transfer overhead on Apple Silicon
2. **Native Metal**: Full GPU acceleration via MLX (vs CPU-only for PyKAN on macOS)
3. **Lazy Evaluation**: MLX's computation graph optimization
4. **Batch-grad Trick**: Efficient per-sample gradients for PINNs

**Run your own benchmarks:**
```bash
pip install -e ".[benchmark]"
python benchmarks/compare_pykan.py
```

*Performance varies by hardware, model size, and workload.*

### Unique Physics Features

```python
from kan_mlx_physics.pde import (
    PDEBuilder, PDEResidualLoss, BoundaryConditionLoss,
    NormalizationLoss, NonTrivialLoss, EigenvalueLoss,
)

# Solve eigenvalue problems with trainable parameters
L = 2.0  # Box length
model, history = (
    PDEBuilder("Derivative(psi, x, 2)/2 + E*psi = 0")
    .params(E=1.0)
    .trainable_params("E")  # Train eigenvalue via Adam
    .domain([0, L])
    .loss(PDEResidualLoss(weight=500))
    .loss(BoundaryConditionLoss(bc_type="dirichlet", weight=1000))
    .loss(NormalizationLoss(weight=100))
    .loss(NonTrivialLoss(weight=200))
    .loss(EigenvalueLoss(param_name="E", method="trainable", weight=3))
    .phase("initial", steps=2000, lr=0.003)
    .phase("refine", steps=1000, lr=0.0015, grid_update_before=True)
    .model(width=[1, 2, 1], grid=5, k=3)
    .solve(verbose=True)
)

# Access trained eigenvalue
E_trained = history.trainable_params["E"]
print(f"E = {E_trained:.6f}")  # E ≈ 1.2337 (analytic: π²/8)
```

#### Operator-identity supervision (data-free discovery)

For phase-space (Wigner/Moyal) eigenvalue problems, the library supports **data-free
supervision by operator identities** — physics losses whose target values are fixed by
universal quantum-mechanical law rather than by the (unknown) analytic solution. This
lets you *discover* an eigenstate from the star-genvalue equation alone, with no
supervised target:

- **Trace** `Tr ρ = 1` — normalization.
- **Purity** `Tr ρ² = 1` — the pure-state condition (`2π² ∫ W² ds = 1` in phase space).
- **Energy variance** `⟨H²⟩ − ⟨H⟩² = 0` — the defining signature of an energy
  eigenstate (via the exact Moyal relation `H ⋆ H = H² − ℏ²/4` for a quadratic `H`).
- **Hilbert–Schmidt deflation** `Tr(ρₙρₘ) = 0` — orthogonality to already-recovered
  lower states, for progressive excited-state extraction.

Each is a subclass of `LossTerm`, composes with the PDE residual and decay/boundary
terms, and injects **no** information about the analytic solution — so the same
machinery that solves the harmonic-oscillator Wigner function extends to problems
where the closed form is unknown or exotic (e.g. Meijer G-function phase-space
distributions in anisotropic quantum cosmology).

---

## Architecture

KAN-MLX-Physics is organized into modular layers:

### Core KAN Engine
```
kan_mlx_physics/
├── multkan.py          # MultKAN model class
├── kan_layer.py        # Individual KAN layers with pluggable basis
├── convenience.py      # High-level API: quick_fit, auto_formula, visualize
├── presets.py          # Physics presets: quantum_oscillator, wave_equation, etc.
├── spline.py           # B-spline basis functions (legacy)
├── symbolic.py         # 25+ symbolic functions + regression
├── functional.py       # Pure functional API + optimizers
├── visualization.py    # PyKAN-style network plots
└── utils.py            # Dataset utilities
```

### Pluggable Basis Module
```
kan_mlx_physics/basis/
├── base.py             # Basis abstract class + utilities
├── bspline.py          # B-spline basis (default)
├── fourier.py          # Fourier basis (periodic functions)
├── chebyshev.py        # Chebyshev polynomials (spectral methods)
├── hermite.py          # Hermite polynomials (quantum mechanics)
├── laguerre.py         # Laguerre polynomials (radial problems)
├── legendre.py         # Legendre polynomials (angular momentum)
└── registry.py         # make_basis(), list_bases(), recommend_basis()
```

### Physics Add-on
```
kan_mlx_physics/
├── physics_symbolic.py # 50+ physics functions (Hermite, Bessel, etc.)
├── pinn.py             # PINN utilities (batch-grad sum trick)
├── formula_render.py   # Multi-format formula output
└── pde/                # PDE solver DSL
    ├── dsl.py          # PDEBuilder fluent interface
    ├── trainer.py      # PDETrainer with trainable params
    ├── losses.py       # Loss terms (PDE, BC, normalization, etc.)
    ├── compiler.py     # Equation parser
    ├── operators.py    # Differential operators, Moyal bracket
    └── physics.py      # Pre-built physics problems
```

---

## Documentation

| Guide | Audience |
|-------|----------|
| [Quick Start](docs/QUICKSTART.md) | Beginners - First model in 10 minutes |
| [API Reference](docs/API_REFERENCE.md) | ML/MLX Developers - Full API docs |
| [Physics Guide](docs/PHYSICS_GUIDE.md) | Scientists - PDE solving, PINNs |
| [PyKAN Migration](docs/PYKAN_MIGRATION.md) | PyKAN Users - Side-by-side comparison |
| [Contributing](CONTRIBUTING.md) | Contributors - Development setup |

---

## Core Concepts

### What is a KAN?

Kolmogorov-Arnold Networks replace fixed activation functions (ReLU, tanh) with **learnable univariate functions** on edges. Based on the [Kolmogorov-Arnold representation theorem](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Arnold_representation_theorem):

```
f(x_1, ..., x_n) = sum_i Phi_i(sum_j phi_ij(x_j))
```

Each phi_ij is a learnable function (B-spline, Fourier, or other basis), making KANs:
- **Interpretable** - Extract symbolic formulas after training
- **Efficient** - Fewer parameters for smooth functions
- **Physics-friendly** - Natural for differential equations
- **Flexible** - Choose the optimal basis for your problem

### Key Components

```python
from kan_mlx_physics import MultKAN

# Width: [input_dim, hidden1, hidden2, ..., output_dim]
# Grid: Number of spline intervals (higher = more expressive)
# k: Spline order (3 = cubic, most common)
model = MultKAN(
    width=[2, 5, 5, 1],    # 2 inputs -> 5 -> 5 -> 1 output
    grid=5,                 # 5 spline intervals
    k=3,                    # Cubic B-splines
)

# Or use physics-optimized basis functions
model = MultKAN(
    width=[2, 10, 1],
    basis="fourier",        # Fourier basis for periodic functions
    basis_M=11,             # 11 basis functions (5 harmonics)
)
```

### Choosing a Basis

| Basis | Best For | Domain | Example Use |
|-------|----------|--------|-------------|
| `bspline` | General (default) | Adaptive | Function approximation |
| `fourier` | Periodic/oscillatory | ℝ | Wave equations, Bloch functions |
| `chebyshev` | Spectral methods | [-1, 1] | Bounded intervals, high accuracy |
| `hermite` | Quantum mechanics | ℝ | Harmonic oscillator wavefunctions |
| `laguerre` | Radial problems | [0, ∞) | Hydrogen atom, radial Schrödinger |
| `legendre` | Angular momentum | [-1, 1] | Spherical harmonics, Legendre ODEs |

```python
from kan_mlx_physics.basis import recommend_basis

# Get recommendations for your problem
bases = recommend_basis("quantum_mechanics", "bounded")
# Returns: ['hermite', 'chebyshev', 'legendre']
```

---

## Advanced Usage

### Physics-Informed Neural Networks (PINNs)

The batch-grad sum trick enables efficient derivative computation:

```python
from kan_mlx_physics import MultKAN, get_params_list
from kan_mlx_physics.pinn import make_derivative_fns
from kan_mlx_physics.functional import functional_forward

model = MultKAN(width=[1, 10, 1], grid=5, k=3)
params = get_params_list(model)

# Get derivative functions
u_fn, du_dx, d2u_dx2 = make_derivative_fns(model.k, model.base_fun)

# Define PDE loss (e.g., harmonic oscillator: u'' + u = 0)
def pde_loss(params, x):
    u = u_fn(params, x)
    u_xx = d2u_dx2(params, x)
    return mx.mean((u_xx + u)**2)
```

### Multiplication Nodes

For learning products like f(x,y) = x * y:

```python
# Extended width format: [[n_sum, n_mult], ...]
model = MultKAN(
    width=[[2, 0], [4, 1], [1, 0]],  # 1 mult node in hidden layer
    mult_arity=2,                      # Each mult node takes 2 inputs
)
```

### L-BFGS Optimization

Faster convergence for physics problems:

```python
from kan_mlx_physics.functional import lbfgs_fit

model, result = lbfgs_fit(
    model,
    dataset,
    max_iter=100,
    tolerance_grad=1e-7
)
```

### Symbolic Regression

```python
# Auto-detect symbolic forms
model.auto_symbolic(x_sample, r2_threshold=0.95)

# Get formula in multiple formats
print(model.symbolic_formula(['x', 'y']))           # Unicode: sin(x) + y^2
print(model.symbolic_formula_latex(['x', 'y']))     # LaTeX
print(model.symbolic_formula_typst(['x', 'y']))     # Typst
sympy_expr = model.to_sympy(['x', 'y'])             # SymPy expression
```

### Physics Symbolic Functions

```python
from kan_mlx_physics.physics_symbolic import register_physics_symbolic, list_physics_symbolic

# Register 50+ physics functions
register_physics_symbolic()

# Available categories:
# - Quantum Mechanics: H_0..H_4, psi_0..psi_2, L_0..L_2
# - Special Functions: J_0, J_1, erf, gamma, Ai, Bi
# - QFT: propagator, bose, fermi, planck
# - Cosmology: a_matter, a_rad, schwarzschild
# - Deformation: q_exp, moyal_1, moyal_2

print(list_physics_symbolic())
```

### Network Pruning

Remove low-importance edges and nodes to simplify the network:

```python
# Train model first
model.fit(dataset, steps=500)

# Auto-prune: remove edges and nodes below threshold
model.prune(threshold=0.01)

# Or prune selectively:
model.prune_edges(threshold=0.01)    # Remove weak edges only
model.prune_nodes(threshold=0.01)    # Remove inactive hidden nodes
model.prune_input(threshold=0.01)    # Remove unused input dimensions

# Manual pruning with specific neuron IDs
model.prune(mode="manual", active_neurons_id=[[0, 1], [0, 2, 3]])
```

Pruning benefits:
- Reduces model complexity for interpretability
- Speeds up inference after training
- Reveals important input features via `prune_input()`

---

## Visualization

```python
# PyKAN-style network diagram
model.plot(folder="./figures", beta=3.0)
```

Produces vertical layout with:
- Activation functions as inset plots on edges
- Transparency based on edge importance
- Sum symbols at nodes
- Red edges for symbolic, black for splines

---

## Comparison with PyKAN

### Feature Matrix

| Feature | KAN-MLX-Physics | PyKAN |
|---------|-----------------|-------|
| **Framework** | MLX (Apple Silicon native) | PyTorch (CUDA/CPU) |
| **Basis Functions** | 6 types (B-spline, Fourier, Chebyshev, Hermite, Laguerre, Legendre) | B-splines only |
| **Per-layer Basis** | Yes | No |
| **PINN Support** | Built-in (PINNTrainer, batch-grad trick) | Manual |
| **PDE DSL** | Yes (PDEBuilder with trainable params) | No |
| **Trainable Eigenvalues** | Built-in (.trainable_params()) | Manual |
| **Symbolic Functions** | 50+ (physics-focused) | ~20 |
| **Formula Output** | 4 formats (Unicode, LaTeX, Typst, SymPy) | 2 formats |
| **Functional API** | Full (stateless, mx.grad compatible) | Partial |
| **Optimizers** | Adam, AdamW, SGD, L-BFGS | Adam, L-BFGS |
| **Pruning** | Yes (`prune`, `prune_edges`, `prune_nodes`, `prune_input`) | Yes |
| **Grid Refinement** | Yes | Yes |
| **Platform** | macOS (Apple Silicon) | Cross-platform |

### When to choose this library

**Choose KAN-MLX-Physics if:**
- You are on Apple Silicon (M1/M2/M3/M4) and want native Metal acceleration
- You need physics-optimized bases (Hermite, Laguerre, Chebyshev, Fourier, …)
- You are solving or *discovering* PDE eigenproblems with the physics DSL
- You want symbolic recovery against a physics-motivated vocabulary

**Choose PyKAN (or a JAX/PyTorch KAN) if:**
- You need cross-platform / CUDA support or must reproduce on non-Apple hardware
- Your losses are dominated by second-order derivatives at scale (see the
  autodiff caveat above)
- You want the larger, more mature community and ecosystem

---

## Citation

If you use KAN-MLX-Physics in your research, please cite **the software** (a
`CITATION.cff` is provided, and GitHub's "Cite this repository" button reads it):

```bibtex
@software{martinez_kan_mlx_physics,
  author  = {Mart\'inez Trejo, Yael Tonatiuh},
  title   = {{KAN-MLX-Physics}: Physics-First Kolmogorov--Arnold Networks
             for Apple Silicon},
  year    = {2026},
  url     = {https://github.com/yaelmartinez/kan-mlx-physics},
  note    = {Version 0.1.0}
}
```

Please also cite **the original KAN paper**:

```bibtex
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and
          Halverson, James and Solja{\v{c}}i{\'c}, Marin and Hou, Thomas Y and
          Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [PyKAN](https://github.com/KindXiaoming/pykan) - Original KAN implementation (MIT License)
- [MLX](https://github.com/ml-explore/mlx) - Apple's machine learning framework
- Inspired by physics research in quantum cosmology and deformation quantization
