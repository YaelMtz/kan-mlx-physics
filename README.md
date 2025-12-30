# KAN-MLX-Physics

**Physics-First Kolmogorov-Arnold Networks for Apple Silicon**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-0.21+-orange.svg)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

KAN-MLX-Physics is a high-performance implementation of [Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) optimized for Apple Silicon. Built on [MLX](https://github.com/ml-explore/mlx), it provides specialized tools for physics-informed neural networks (PINNs) and PDE solving.

> **Note:** This is an independent implementation inspired by [PyKAN](https://github.com/KindXiaoming/pykan). It is not affiliated with the existing `mlx-kan` package on PyPI.

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Apple Silicon Optimized** | Native Metal acceleration via MLX, reduced memory overhead with unified memory |
| **Pluggable Basis Functions** | B-splines, Fourier, Chebyshev, Hermite, Laguerre, Legendre - choose the right basis for your physics |
| **Physics-First Design** | PDE DSL, 50+ physics symbolic functions, Moyal star product |
| **Efficient Derivatives** | Batch-grad sum trick for PINN training |
| **PyKAN Compatible** | Similar API patterns, identical visualization style |
| **Functional API** | Pure functions for `mx.grad()`, stateless parameter handling |
| **Multi-Format Output** | Formula export to Unicode, LaTeX, Typst, SymPy |

---

## Quick Install

```bash
# From source (recommended)
git clone https://github.com/your-username/kan-mlx-physics.git
cd kan-mlx-physics
pip install -e "."

# With optional dependencies
pip install -e ".[dev]"      # Development tools
pip install -e ".[viz]"      # Visualization (matplotlib)
pip install -e ".[physics]"  # Physics extras (sympy)
pip install -e ".[all]"      # Everything
```

**Requirements:** macOS with Apple Silicon (M1/M2/M3/M4), Python 3.10+, MLX 0.21+

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

### Benchmarks: KAN-MLX-Physics vs PyKAN

> **Methodology:** Side-by-side comparison on M3 Max (36GB unified memory), macOS Tahoe 26.2, Python 3.11.
> Model: `width=[2, 5, 1]`, `grid=5`, `k=3`, batch size 1000.
> PyKAN on CPU backend (see MPS note below), KAN-MLX-Physics on Metal via MLX.
> See [`benchmarks/compare_pykan.py`](benchmarks/compare_pykan.py) to reproduce on your hardware.

| Benchmark | KAN-MLX-Physics | PyKAN (CPU) | Speedup |
|-----------|-----------------|-------------|---------|
| Forward Pass | 1.0 ms | 5.7 ms | **5.9x** |
| Gradient Computation | 1.9 ms | 14.3 ms | **7.7x** |
| Training (2000 steps) | 3.9 s | 13.9 s | **3.5x** |

> **Note on Symbolic Regression:** PyKAN's symbolic regression uses a different methodology (exhaustive library search) compared to KAN-MLX-Physics (R² fitting), making direct timing comparisons not meaningful. Both achieve accurate symbolic formula extraction.

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
from kan_mlx_physics.pde import solve

# Solve Schrodinger equation
psi, history = solve(
    "-nabla^2 psi/2 + x^2 psi/2 = E psi",
    domain=[-5, 5],
    params={"E": 0.5}
)

# Wheeler-DeWitt with Moyal deformation (quantum cosmology)
Psi, _ = solve(
    "H star_theta Psi = 0",
    domain=[0.1, 5],
    params={"theta": 0.05}
)
```

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
    ├── dsl.py          # Equation parser
    ├── operators.py    # Differential operators, Moyal bracket
    ├── physics.py      # Pre-built physics problems
    └── solver.py       # Solver configuration
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

| Aspect | KAN-MLX-Physics | PyKAN |
|--------|-----------------|-------|
| **Backend** | MLX (Apple Silicon) | PyTorch (CUDA/CPU) |
| **Functional API** | Native | Partial |
| **Physics Functions** | 50+ | ~20 |
| **Formula Output** | 4 formats | 2 formats |
| **Platform** | macOS (Apple Silicon) | Cross-platform |
| **Ecosystem** | Research-focused | Mature community |

**Choose KAN-MLX-Physics if:** You're on Apple Silicon and need physics-informed training with efficient derivatives.

**Choose PyKAN if:** You need cross-platform support or extensive community resources.

---

## Citation

If you use KAN-MLX-Physics in your research, please cite the original KAN paper:

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
