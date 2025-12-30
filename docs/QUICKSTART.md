# Quick Start Guide

Get up and running with KAN-MLX-Physics in 10 minutes.

---

## Table of Contents

1. [Installation](#installation)
2. [Your First KAN](#your-first-kan)
3. [Understanding the Architecture](#understanding-the-architecture)
4. [Training Basics](#training-basics)
5. [Visualization](#visualization)
6. [Symbolic Regression](#symbolic-regression)
7. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

- **macOS** with Apple Silicon M series chips
- **Python 3.11+**
- **MLX 0.21+** (installed automatically)

### Install Methods

```bash
# From source (recommended)
git clone https://github.com/yaelmartinez/kan-mlx-physics.git
cd kan-mlx-physics
pip install -e "."

# With optional dependencies
pip install -e ".[dev]"      # Development tools
pip install -e ".[viz]"      # Visualization (matplotlib)
pip install -e ".[physics]"  # Physics extras (sympy)
pip install -e ".[all]"      # Everything
```

### Verify Installation

```python
import mlx.core as mx
from kan_mlx_physics import MultKAN

model = MultKAN(width=[2, 3, 1])
x = mx.random.uniform(shape=(10, 2))
y = model(x)
print(f"Output shape: {y.shape}")  # (10, 1)
print("KAN-MLX-Physics installed successfully!")
```

---

## Your First KAN

### Quick Way (Recommended)

Use `quick_fit` for a one-liner experience:

```python
from kan_mlx_physics import quick_fit, auto_formula

# One-liner: create, train, and get model
model, history = quick_fit(
    f=lambda x: mx.sin(mx.pi * x[:, 0]) + x[:, 1]**2,
    n_var=2,
    steps=500,
)

# Extract symbolic formula
formula = auto_formula(model, history['x_sample'], ['x', 'y'])
print(formula)  # sin(3.14*x) + y^2
```

### Verbose Way (Full Control)

Let's learn the function f(x, y) = sin(pi * x) + y^2:

```python
import mlx.core as mx
from kan_mlx_physics import MultKAN, create_dataset

# Step 1: Create training data
def target_function(x):
    return mx.sin(mx.pi * x[:, 0]) + x[:, 1]**2

dataset = create_dataset(
    f=target_function,
    n_var=2,              # 2 input variables (x, y)
    train_num=1000,       # 1000 training samples
    test_num=200,         # 200 test samples
    ranges=(-1, 1),       # Sample from [-1, 1]
)

# Step 2: Create the model
model = MultKAN(
    width=[2, 5, 1],      # 2 inputs -> 5 hidden -> 1 output
    grid_points=5,        # 5 spline grid intervals (alias: grid)
    spline_order=3,       # Cubic splines (alias: k)
)

# Step 3: Train
history = model.fit(
    dataset,
    steps=500,
    learning_rate=0.01,   # alias: lr
    regularization=0.01,  # alias: lamb
)

# Step 4: Check accuracy
from kan_mlx_physics.functional import functional_forward, get_params_list

params = get_params_list(model)
x_test = dataset['test_input']
y_test = dataset['test_label']
y_pred = functional_forward(params, x_test, model.k, model.base_fun)

mse = mx.mean((y_pred - y_test)**2)
print(f"Test MSE: {float(mse):.6f}")
```

---

## Understanding the Architecture

### What Makes KANs Different?

**Traditional Neural Networks (MLPs):**
```
input -> [linear] -> [ReLU] -> [linear] -> [ReLU] -> output
              ^           ^
          weights    fixed activation
```

**Kolmogorov-Arnold Networks (KANs):**
```
input -> [learnable spline edges] -> [sum] -> [learnable splines] -> output
                   ^
         activation IS the weight
```

### Width Parameter

The `width` list defines your network architecture:

```python
width = [2, 5, 3, 1]
#        ^  ^  ^  ^
#        |  |  |  +-- 1 output
#        |  |  +-- 3 nodes in hidden layer 2
#        |  +-- 5 nodes in hidden layer 1
#        +-- 2 inputs

# Number of edges = 2*5 + 5*3 + 3*1 = 28 learnable functions
```

### Grid and Spline Order

```python
model = MultKAN(
    width=[2, 5, 1],
    grid=5,      # More grid points = more expressive splines
    k=3,         # k=3 is cubic (smooth), k=1 is linear
)
```

| Grid | Expressiveness | Training Speed |
|------|---------------|----------------|
| 3 | Low | Fast |
| 5 | Medium (default) | Balanced |
| 10 | High | Slower |
| 20+ | Very High | Slow, risk overfitting |

### Physics Presets

Use presets for common physics use cases:

```python
from kan_mlx_physics import from_preset, list_presets

# See available presets
print(list_presets())
# {'quantum_oscillator': 'Hermite basis for QM harmonic oscillator',
#  'wave_equation': 'Fourier basis for periodic/wave problems',
#  'spectral': 'Chebyshev basis for high-accuracy spectral methods',
#  'radial': 'Laguerre basis for radial problems (hydrogen atom)',
#  'angular': 'Legendre basis for angular momentum',
#  'general': 'B-spline basis for general function approximation'}

# Create model from preset
model = from_preset("quantum_oscillator", width=[1, 10, 1])
model = from_preset("wave_equation", width=[2, 15, 1])

# quick_fit also supports presets
model, history = quick_fit(
    f=my_oscillator,
    n_var=1,
    preset="quantum_oscillator",
)
```

### Basis Functions

Choose the right basis for your problem:

```python
# Default B-spline (general purpose)
model = MultKAN(width=[2, 5, 1])

# Fourier for periodic functions (waves, oscillations)
model = MultKAN(width=[2, 10, 1], basis="fourier", basis_M=11)

# Chebyshev for spectral methods (bounded domains)
model = MultKAN(width=[2, 10, 1], basis="chebyshev", basis_M=8)

# Hermite for quantum mechanics (harmonic oscillator)
model = MultKAN(width=[1, 10, 1], basis="hermite", basis_kwargs={"weighted": True})

# Laguerre for radial problems (hydrogen atom)
model = MultKAN(width=[1, 10, 1], basis="laguerre", basis_kwargs={"alpha": 1.0})
```

| Basis | Best For | Example |
|-------|----------|---------|
| `bspline` | General (default) | Any smooth function |
| `fourier` | Periodic/oscillatory | sin(x), wave equations |
| `chebyshev` | Bounded intervals | Spectral methods |
| `hermite` | Quantum mechanics | Harmonic oscillator |
| `laguerre` | Radial problems | Hydrogen wavefunctions |
| `legendre` | Angular momentum | Spherical harmonics |

---

## Training Basics

### The fit() Method

```python
history = model.fit(
    dataset,

    # Optimization
    opt="Adam",           # "Adam", "AdamW", "SGD", or "LBFGS"
    steps=1000,           # Number of training steps
    lr=0.01,              # Learning rate
    batch_size=-1,        # -1 = full batch

    # Regularization
    lamb=0.01,            # Overall regularization strength
    lamb_l1=1.0,          # L1 on spline coefficients
    lamb_entropy=2.0,     # Entropy regularization (sparsity)

    # Grid updates
    update_grid=True,     # Adapt grid to data distribution
    grid_update_freq=100, # How often to update

    # Logging
    log=10,               # Print every 10 steps
)
```

### Training History

```python
# Access training metrics
print(history['train_loss'])    # List of training losses
print(history['test_loss'])     # List of test losses (if test data provided)
print(history['reg'])           # Regularization values

# Plot training curve
import matplotlib.pyplot as plt
plt.plot(history['train_loss'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.show()
```

### Using LBFGS for Faster Convergence

For physics problems, L-BFGS often converges faster:

```python
from mlx_kan.functional import lbfgs_fit

model, result = lbfgs_fit(
    model,
    dataset,
    max_iter=100,
    tolerance_grad=1e-7,
    verbose=True,
)

print(f"Final loss: {result.fun:.6f}")
print(f"Converged: {result.success}")
```

---

## Visualization

### Network Diagram

```python
# Create PyKAN-style visualization
model.plot(
    folder="./figures",    # Save location
    beta=3.0,              # Edge transparency scaling
    scale=0.5,             # Figure size scale
    title="My KAN Model",
)
```

This produces:
- Vertical layout (input at bottom)
- Activation functions drawn on edges
- Edge transparency = importance
- Sum symbols at nodes

### Activation Functions

```python
from mlx_kan import plot_activations

# Plot all activation functions in a layer
plot_activations(model, layer_idx=0, x=dataset['train_input'])
```

### Training History

```python
from mlx_kan import plot_training_history

plot_training_history(history, metric='loss')
```

---

## Symbolic Regression

One of KAN's unique features: extract interpretable formulas from trained networks.

### Automatic Discovery

```python
# Run a forward pass to cache activations
x_sample = dataset['train_input']
_ = model(x_sample)

# Auto-detect symbolic functions
fixed_edges = model.auto_symbolic(
    x_sample,
    r2_threshold=0.95,    # Minimum R^2 for a match
)

print(f"Fixed {len(fixed_edges)} edges to symbolic functions")
```

### Get the Formula

```python
# Unicode format (terminal-friendly)
formula = model.symbolic_formula(var_names=['x', 'y'])
print(formula)  # sin(3.14*x) + y^2

# LaTeX format (for papers)
latex = model.symbolic_formula_latex(var_names=['x', 'y'])
print(latex)  # \sin(3.14 \cdot x) + y^{2}

# SymPy expression (for further manipulation)
import sympy
expr = model.to_sympy(var_names=['x', 'y'])
print(sympy.simplify(expr))
```

### Manual Symbolic Assignment

If you know what function an edge should be:

```python
# Fix edge (layer, input_idx, output_idx) to a function
model.fix_symbolic(
    l=0,           # First layer
    i=0,           # First input
    j=0,           # First output in that layer
    fn_name='sin', # Use sine function
)
```

### Available Symbolic Functions

```python
from mlx_kan import list_symbolic

print(list_symbolic())
# ['x', 'x^2', 'x^3', 'x^4', 'x^0.5', 'x^-1', 'x^-2',
#  'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
#  'exp', 'log', 'sinh', 'cosh', 'tanh',
#  'abs', 'sign', 'gaussian', 'sigmoid', 'relu', 'softplus', ...]
```

### Physics Functions

```python
from mlx_kan.physics_symbolic import register_physics_symbolic, list_physics_symbolic

# Add 50+ physics functions
register_physics_symbolic()

# See what's available
physics_fns = list_physics_symbolic()
print(physics_fns['Quantum Mechanics'])
# ['H_0', 'H_1', 'H_2', 'H_3', 'H_4',  # Hermite polynomials
#  'psi_0', 'psi_1', 'psi_2',           # Harmonic oscillator wavefunctions
#  'L_0', 'L_1', 'L_2',                 # Laguerre polynomials
#  ...]
```

---

## Next Steps

### For ML Developers

- Read [API Reference](API_REFERENCE.md) for full documentation
- Explore the functional API in `mlx_kan.functional`
- Try multiplication nodes for learning products

### For Scientists/Physicists

- Read [Physics Guide](PHYSICS_GUIDE.md) for PDE solving
- Learn the batch-grad sum trick for efficient derivatives
- Use the PDE DSL for quick equation solving

### For PyKAN Users

- Read [PyKAN Migration Guide](PYKAN_MIGRATION.md)
- Note: Most PyKAN code works with minimal changes
- Key difference: Use `mx.array` instead of `torch.Tensor`

---

## Common Issues

### "No module named 'mlx'"

MLX only works on Apple Silicon Macs. Check your hardware:
```bash
uname -m  # Should show "arm64"
```

### Slow Training

1. Reduce grid size: `grid=3` instead of `grid=10`
2. Use smaller batches: `batch_size=256`
3. Use `@mx.compile` for custom training loops

### NaN in Training

1. Reduce learning rate: `lr=0.001`
2. Add regularization: `lamb=0.1`
3. Check your data for outliers

### Symbolic Regression Fails

1. Train longer before calling `auto_symbolic()`
2. Lower threshold: `r2_threshold=0.9`
3. Try manual assignment with `fix_symbolic()`

---

## Example: Complete Workflow

```python
import mlx.core as mx
from mlx_kan import MultKAN, create_dataset, plot_training_history

# 1. Data
dataset = create_dataset(
    f=lambda x: mx.sin(mx.pi * x[:, 0]) * mx.exp(-x[:, 1]**2),
    n_var=2,
    train_num=2000,
)

# 2. Model
model = MultKAN(width=[2, 8, 1], grid=5, k=3)

# 3. Train
history = model.fit(
    dataset,
    steps=1000,
    lr=0.01,
    lamb=0.01,
    lamb_entropy=2.0,
    log=100,
)

# 4. Visualize training
plot_training_history(history)

# 5. Prune unimportant edges
model.prune(threshold=0.01)

# 6. Refine grid
model.refine(new_grid=10)

# 7. Train more
history2 = model.fit(dataset, steps=500, lr=0.005)

# 8. Extract formula
model.auto_symbolic(dataset['train_input'])
formula = model.symbolic_formula(['x', 'y'])
print(f"Discovered formula: {formula}")

# 9. Visualize final model
model.plot(title="Final Model")
```

---

Now you're ready to use MLX-KAN! Check the other guides for more advanced topics.
