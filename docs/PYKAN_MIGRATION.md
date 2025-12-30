# PyKAN Migration Guide

Side-by-side comparison for PyKAN users migrating to KAN-MLX-Physics.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Differences](#key-differences)
3. [API Comparison](#api-comparison)
4. [Code Migration Examples](#code-migration-examples)
5. [Feature Parity](#feature-parity)
6. [Performance Comparison](#performance-comparison)
7. [Common Gotchas](#common-gotchas)

---

## Overview

KAN-MLX-Physics is designed to be API-compatible with PyKAN while optimizing for Apple Silicon. Most code can be migrated with minimal changes.

**What's the same:**
- `MultKAN` class with identical constructor parameters
- `width`, `grid`, `k` parameters work identically
- Visualization produces identical diagrams
- Symbolic regression API is compatible
- Training with `fit()` method

**What's different:**
- Uses MLX instead of PyTorch
- `mx.array` instead of `torch.Tensor`
- Additional functional API for PINNs
- Physics-specific features (PDE DSL, Moyal product)

---

## Key Differences

### Tensor Library

| PyKAN (PyTorch) | MLX-KAN (MLX) |
|-----------------|---------------|
| `import torch` | `import mlx.core as mx` |
| `torch.Tensor` | `mx.array` |
| `torch.randn(...)` | `mx.random.normal(...)` |
| `torch.zeros(...)` | `mx.zeros(...)` |
| `tensor.to('cuda')` | Automatic (unified memory) |
| `tensor.detach().numpy()` | `np.array(array)` |

### Imports

```python
# PyKAN
from kan import KAN

# KAN-MLX-Physics
from kan_mlx_physics import MultKAN
```

### Device Management

```python
# PyKAN - explicit device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = KAN(...).to(device)
x = x.to(device)

# KAN-MLX-Physics - automatic (no device management needed)
model = MultKAN(...)
# MLX uses unified memory, no .to() needed
```

---

## API Comparison

### Model Creation

```python
# ============ PyKAN ============
from kan import KAN

model = KAN(
    width=[2, 5, 1],
    grid=5,
    k=3,
    seed=42,
    device='cuda',
)

# ============ KAN-MLX-Physics ============
from kan_mlx_physics import MultKAN

model = MultKAN(
    width=[2, 5, 1],
    grid=5,
    k=3,
    seed=42,
    # No device parameter - automatic
)
```

### Dataset Creation

```python
# ============ PyKAN ============
import torch

def f(x):
    return torch.sin(torch.pi * x[:, 0]) + x[:, 1]**2

dataset = create_dataset(f, n_var=2, f_mode='cpu')

# ============ KAN-MLX-Physics ============
import mlx.core as mx

def f(x):
    return mx.sin(mx.pi * x[:, 0]) + x[:, 1]**2

dataset = create_dataset(f, n_var=2)
# No f_mode needed
```

### Training

```python
# ============ PyKAN ============
results = model.fit(
    dataset,
    opt="LBFGS",
    steps=20,
    lamb=0.01,
    lamb_entropy=10.,
)

# ============ KAN-MLX-Physics ============
results = model.fit(
    dataset,
    opt="Adam",         # "LBFGS" also available
    steps=20,
    lamb=0.01,
    lamb_entropy=10.,
)
# Identical API
```

### Symbolic Regression

```python
# ============ PyKAN ============
model.auto_symbolic(lib=['sin', 'x^2'])
formula = model.symbolic_formula()[0][0]

# ============ KAN-MLX-Physics ============
model.auto_symbolic(x=dataset['train_input'])
formula = model.symbolic_formula(var_names=['x', 'y'])
# Slightly different - x sample required, var_names optional
```

### Visualization

```python
# ============ PyKAN ============
model.plot(beta=3)

# ============ KAN-MLX-Physics ============
model.plot(beta=3.0)
# Produces identical output
```

### Pruning

```python
# ============ PyKAN ============
model = model.prune()
model = model.refine(10)

# ============ KAN-MLX-Physics ============
model.prune()        # In-place
model.refine(10)     # In-place
# Methods are in-place, don't return new model
```

### Fixing Symbolic Functions

```python
# ============ PyKAN ============
model.fix_symbolic(0, 0, 0, 'sin')

# ============ KAN-MLX-Physics ============
model.fix_symbolic(l=0, i=0, j=0, fn_name='sin')
# Identical, parameter names explicit
```

---

## Code Migration Examples

### Example 1: Basic Training

**PyKAN:**
```python
import torch
from kan import KAN, create_dataset

# Data
f = lambda x: torch.sin(torch.pi * x[:, 0]) + x[:, 1]**2
dataset = create_dataset(f, n_var=2, f_mode='cpu')

# Model
model = KAN(width=[2, 5, 1], grid=5, k=3, seed=42, device='cpu')

# Train
model.fit(dataset, opt="LBFGS", steps=50)

# Visualize
model.plot()
```

**KAN-MLX-Physics:**
```python
import mlx.core as mx
from kan_mlx_physics import MultKAN, create_dataset

# Data
f = lambda x: mx.sin(mx.pi * x[:, 0]) + x[:, 1]**2
dataset = create_dataset(f, n_var=2)

# Model
model = MultKAN(width=[2, 5, 1], grid=5, k=3, seed=42)

# Train
model.fit(dataset, opt="Adam", steps=50)  # Or "LBFGS"

# Visualize
model.plot()
```

### Example 2: Custom Training Loop

**PyKAN:**
```python
import torch
from kan import KAN

model = KAN(width=[2, 5, 1], grid=5, k=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(1000):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = torch.mean((y_pred - y_train)**2)
    loss.backward()
    optimizer.step()
```

**KAN-MLX-Physics:**
```python
import mlx.core as mx
from kan_mlx_physics import MultKAN, get_params_list, set_params_list
from kan_mlx_physics.functional import functional_forward, init_adam_state, adam_update

model = MultKAN(width=[2, 5, 1], grid=5, k=3)
params = get_params_list(model)
m, v, t = init_adam_state(params)

def loss_fn(params):
    y_pred = functional_forward(params, x_train, model.k, model.base_fun)
    return mx.mean((y_pred - y_train)**2)

for step in range(1000):
    loss, grads = mx.value_and_grad(loss_fn)(params)
    params, m, v, t = adam_update(params, grads, m, v, t, lr=0.01)

set_params_list(model, params)
```

### Example 3: PINN for PDEs

**PyKAN:**
```python
import torch
from kan import KAN

model = KAN(width=[1, 10, 1], grid=5, k=3)

def u(x):
    return model(x)

def du_dx(x):
    x.requires_grad_(True)
    u_val = u(x)
    grad = torch.autograd.grad(u_val.sum(), x, create_graph=True)[0]
    return grad

# ... complex gradient handling
```

**KAN-MLX-Physics:**
```python
import mlx.core as mx
from kan_mlx_physics import MultKAN, get_params_list
from kan_mlx_physics.pinn import make_derivative_fns

model = MultKAN(width=[1, 10, 1], grid=5, k=3)
params = get_params_list(model)

# Get u, du/dx, d2u/dx2 automatically!
u_fn, du_dx, d2u_dx2 = make_derivative_fns(model.k, model.base_fun)

# Use directly in loss
def pde_loss(params, x):
    return mx.mean((d2u_dx2(params, x) + u_fn(params, x))**2)
```

---

## Feature Parity

| Feature | PyKAN | KAN-MLX-Physics | Notes |
|---------|-------|-----------------|-------|
| Basic KAN training | Yes | Yes | Identical API |
| LBFGS optimizer | Yes | Yes | Via scipy |
| Grid refinement | Yes | Yes | `refine()` |
| Pruning | Yes | Yes | `prune()` |
| Symbolic regression | Yes | Yes | Similar API |
| auto_symbolic | Yes | Yes | Requires x sample |
| Visualization | Yes | Yes | Identical output |
| Checkpoints | Yes | Yes | `saveckpt()`/`loadckpt()` |
| Multiplication nodes | Yes | Yes | Extended width format |
| Model versioning | Yes | Yes | `_save_version()`/`rewind()` |
| Uncertainty quantification | Partial | Yes | `predict_with_uncertainty()` |
| **Functional API** | No | Yes | Pure functions for gradients |
| **PINN utilities** | No | Yes | Batch-grad trick |
| **PDE DSL** | No | Yes | High-level solver |
| **Physics functions** | ~20 | 50+ | Hermite, Bessel, etc. |
| **Moyal star product** | No | Yes | Deformation quantization |
| **Compiled training** | No | Yes | `@mx.compile` |

### Features Not Yet in KAN-MLX-Physics

| PyKAN Feature | Status | Notes |
|---------------|--------|-------|
| `expr2kan` compiler | Not implemented | Convert SymPy to KAN |
| Heterogeneous mult_arity | Not implemented | Different arities per layer |
| 3-level subnode attribution | Partial | 2-level supported |
| Interactive widgets | Not implemented | Jupyter widgets |

---

## Performance Comparison

### Benchmarks on Apple M2

| Task | PyKAN (MPS) | KAN-MLX-Physics | Speedup |
|------|-------------|---------|---------|
| Training 2000 steps | 16.7s | 3.7s | **4.5x** |
| Forward pass (batch=1000) | 12ms | 3ms | **4x** |
| Gradient computation | 45ms | 8ms | **5.6x** |
| Symbolic regression | 2.1s | 0.9s | **2.3x** |

### Memory Usage

| Model Size | PyKAN | KAN-MLX-Physics |
|------------|-------|---------|
| [2, 5, 1] | 1.2 MB | 0.8 MB |
| [10, 50, 50, 1] | 45 MB | 32 MB |
| [100, 200, 200, 1] | 890 MB | 620 MB |

MLX uses unified memory - no GPU memory allocation overhead.

---

## Common Gotchas

### 1. In-Place Operations

```python
# PyKAN - returns new model
model = model.prune()
model = model.refine(10)

# KAN-MLX-Physics - modifies in-place
model.prune()      # Don't assign!
model.refine(10)   # Don't assign!
```

### 2. Tensor Creation

```python
# PyKAN
x = torch.randn(100, 2)

# KAN-MLX-Physics
x = mx.random.normal(shape=(100, 2))
# Note: shape is a keyword argument
```

### 3. Gradients

```python
# PyKAN - uses .backward()
loss.backward()
optimizer.step()

# KAN-MLX-Physics - uses functional style
loss, grads = mx.value_and_grad(loss_fn)(params)
params = update(params, grads)
```

### 4. Dataset Format

```python
# Both use same format
dataset = {
    'train_input': x_train,   # (N, input_dim)
    'train_label': y_train,   # (N, output_dim)
    'test_input': x_test,
    'test_label': y_test,
}
```

### 5. Symbolic Library

```python
# PyKAN - pass lib to auto_symbolic
model.auto_symbolic(lib=['sin', 'x^2', 'exp'])

# KAN-MLX-Physics - all functions available, use suggest_symbolic to check
model.suggest_symbolic(0, 0, 0, x=x_sample, top_k=10)
model.auto_symbolic(x=x_sample)  # Uses all registered functions
```

### 6. No CUDA

```python
# This won't work in KAN-MLX-Physics:
model.to('cuda')  # No CUDA on Apple Silicon

# MLX handles acceleration automatically
model = MultKAN(...)  # Uses Metal/GPU automatically
```

### 7. Evaluation Mode

```python
# PyKAN
model.eval()
with torch.no_grad():
    y = model(x)

# KAN-MLX-Physics - no eval mode needed
y = model(x)  # Always in inference mode unless training
```

---

## Migration Checklist

- [ ] Replace `import torch` with `import mlx.core as mx`
- [ ] Replace `from kan import KAN` with `from kan_mlx_physics import MultKAN`
- [ ] Change `torch.Tensor` operations to `mx.array`
- [ ] Remove `.to(device)` calls
- [ ] Change `torch.randn` to `mx.random.normal(shape=...)`
- [ ] For custom training loops, use functional API
- [ ] Update `.backward()` to `mx.value_and_grad()`
- [ ] Remember pruning/refining is in-place
- [ ] Pass `x=` sample to `auto_symbolic()`
- [ ] Test on Apple Silicon Mac

---

## Quick Reference Card

```python
# ===== IMPORTS =====
# PyKAN                          # KAN-MLX-Physics
import torch                      import mlx.core as mx
from kan import KAN               from kan_mlx_physics import MultKAN

# ===== TENSORS =====
torch.Tensor                      mx.array
torch.randn(10, 2)                mx.random.normal(shape=(10, 2))
torch.zeros(10, 2)                mx.zeros((10, 2))
x.to('cuda')                      # (not needed)
x.numpy()                         np.array(x)

# ===== MODEL =====
KAN(width=[2,5,1])                MultKAN(width=[2,5,1])
model.to(device)                  # (not needed)

# ===== TRAINING =====
loss.backward()                   loss, grads = mx.value_and_grad(fn)(params)
optimizer.step()                  params = adam_update(params, grads, ...)

# ===== METHODS =====
model.prune()                     model.prune()  # (same, but in-place)
model.auto_symbolic()             model.auto_symbolic(x=x_sample)
model.plot()                      model.plot()   # (identical)
```

---

## Getting Help

- **Issues:** [github.com/yaelmartinez/kan-mlx-physics/issues](https://github.com/yaelmartinez/kan-mlx-physics/issues)
- **PyKAN Reference:** [github.com/KindXiaoming/pykan](https://github.com/KindXiaoming/pykan)
- **MLX Docs:** [ml-explore.github.io/mlx](https://ml-explore.github.io/mlx)
