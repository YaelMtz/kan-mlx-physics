# API Reference

Complete API documentation for KAN-MLX-Physics.

---

## Table of Contents

1. [Convenience API](#convenience-api) ⭐ Start here!
2. [Presets](#presets)
3. [MultKAN Class](#multkan-class)
4. [KANLayer Class](#kanlayer-class)
5. [Basis Module](#basis-module)
6. [Functional API](#functional-api)
7. [PINN Utilities](#pinn-utilities)
8. [Spline Functions](#spline-functions)
9. [Symbolic Functions](#symbolic-functions)
10. [Visualization](#visualization)
11. [Utilities](#utilities)
12. [PDE Module](#pde-module)

---

## Convenience API

High-level functions for common workflows with minimal boilerplate.

**Location:** `kan_mlx_physics.convenience`

### quick_fit()

```python
quick_fit(
    f: Callable[[mx.array], mx.array],
    n_var: int,
    width: Optional[List[int]] = None,
    preset: Optional[str] = None,
    steps: int = 500,
    train_num: int = 1000,
    test_num: Optional[int] = None,
    domain: Union[Tuple[float, float], List[Tuple[float, float]]] = (-1, 1),
    learning_rate: float = 0.01,
    regularization: float = 0.01,
    verbose: bool = True,
    **model_kwargs
) -> Tuple[MultKAN, Dict[str, Any]]
```

One-liner to create and train a KAN model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `f` | `Callable` | Required | Target function f(x) -> y where x is (batch, n_var) |
| `n_var` | `int` | Required | Number of input variables |
| `width` | `List[int]` | None | Network architecture. If None, uses [n_var, 5*n_var, 1] |
| `preset` | `str` | None | Physics preset name (e.g., "quantum_oscillator") |
| `steps` | `int` | 500 | Number of training steps |
| `train_num` | `int` | 1000 | Number of training samples |
| `test_num` | `int` | None | Number of test samples (default: train_num // 5) |
| `domain` | `Tuple` | `(-1, 1)` | Input domain as (min, max) or per-variable ranges |
| `learning_rate` | `float` | 0.01 | Learning rate |
| `regularization` | `float` | 0.01 | Regularization strength |
| `verbose` | `bool` | True | Print training progress |
| `**model_kwargs` | | | Additional MultKAN arguments |

**Returns:** Tuple of (trained_model, history_dict). History includes `x_sample` for formula extraction.

**Example:**
```python
from kan_mlx_physics import quick_fit

model, history = quick_fit(
    f=lambda x: mx.sin(mx.pi * x[:, 0]) + x[:, 1]**2,
    n_var=2,
    steps=500,
)

# With physics preset
model, history = quick_fit(
    f=harmonic_oscillator,
    n_var=1,
    preset="quantum_oscillator",
)
```

### auto_formula()

```python
auto_formula(
    model: MultKAN,
    x_sample: mx.array,
    var_names: Optional[List[str]] = None,
    threshold: float = 0.95,
    format: str = "unicode",
    verbose: bool = False,
) -> str
```

Extract symbolic formula from a trained model.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `MultKAN` | Required | Trained MultKAN model |
| `x_sample` | `mx.array` | Required | Sample input data for activation caching |
| `var_names` | `List[str]` | None | Variable names. If None, uses ['x_0', 'x_1', ...] |
| `threshold` | `float` | 0.95 | R² threshold for symbolic fitting |
| `format` | `str` | "unicode" | Output format: "unicode", "latex", "typst", "sympy" |
| `verbose` | `bool` | False | Print symbolic fitting details |

**Returns:** Formula string in the requested format.

**Example:**
```python
from kan_mlx_physics import auto_formula

formula = auto_formula(model, history['x_sample'], ['x', 'y'])
print(formula)  # sin(3.14*x) + y^2

latex = auto_formula(model, x, ['x', 'y'], format="latex")
# \sin(3.14 \cdot x) + y^{2}
```

### visualize()

```python
visualize(
    model: MultKAN,
    what: str = "network",
    x: Optional[mx.array] = None,
    **kwargs
) -> None
```

Unified visualization interface for KAN models.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `MultKAN` | Required | Model to visualize |
| `what` | `str` | "network" | What to visualize: "network", "activations", "history" |
| `x` | `mx.array` | None | Input data (required for "activations") |
| `**kwargs` | | | Additional arguments (see below) |

**Keyword Arguments by visualization type:**

- **"network"**: `folder`, `beta`, `scale`, `title`, `in_vars`, `out_vars`
- **"activations"**: `layer` (layer index), `folder`
- **"history"**: `history` (required), `metric` ("loss" or "reg")

**Example:**
```python
from kan_mlx_physics import visualize

# Network diagram
visualize(model, "network", folder="./figs", title="My KAN")

# Activation functions
visualize(model, "activations", x=data, layer=0)

# Training history
visualize(model, "history", history=training_history)
```

### fit_and_extract()

```python
fit_and_extract(
    f: Callable[[mx.array], mx.array],
    n_var: int,
    var_names: Optional[List[str]] = None,
    **kwargs
) -> Tuple[MultKAN, str]
```

Convenience function combining quick_fit and auto_formula.

**Example:**
```python
from kan_mlx_physics import fit_and_extract

model, formula = fit_and_extract(
    f=lambda x: mx.sin(mx.pi * x[:, 0]),
    n_var=1,
    var_names=['x'],
)
print(formula)  # sin(3.14*x)
```

---

## Presets

Physics-optimized model presets for common use cases.

**Location:** `kan_mlx_physics.presets`

### list_presets()

```python
list_presets() -> Dict[str, str]
```

List available presets with descriptions.

**Returns:** Dictionary mapping preset names to descriptions.

**Example:**
```python
from kan_mlx_physics import list_presets

print(list_presets())
# {'quantum_oscillator': 'Quantum harmonic oscillator wavefunctions',
#  'wave_equation': 'Periodic/oscillatory solutions',
#  'spectral': 'High-accuracy spectral methods',
#  'radial': 'Radial problems (hydrogen atom)',
#  'angular': 'Angular momentum, spherical harmonics',
#  'general': 'General-purpose adaptive B-spline'}
```

### from_preset()

```python
from_preset(
    preset: str,
    width: List[int],
    **overrides
) -> MultKAN
```

Create a MultKAN model from a physics preset.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `preset` | `str` | Preset name |
| `width` | `List[int]` | Network architecture |
| `**overrides` | | Override preset parameters |

**Available Presets:**

| Preset | Basis | Best For |
|--------|-------|----------|
| `quantum_oscillator` | Hermite (weighted) | QM harmonic oscillator |
| `wave_equation` | Fourier | Periodic/wave problems |
| `spectral` | Chebyshev | High-accuracy spectral methods |
| `radial` | Laguerre | Radial problems (hydrogen atom) |
| `angular` | Legendre | Angular momentum |
| `general` | B-spline | General function approximation |

**Example:**
```python
from kan_mlx_physics import from_preset

# Quantum oscillator with Hermite basis
model = from_preset("quantum_oscillator", width=[1, 10, 1])

# Wave equation with Fourier basis
model = from_preset("wave_equation", width=[2, 15, 1])

# Override preset parameters
model = from_preset("wave_equation", width=[2, 10, 1], basis_M=31)
```

### describe_preset()

```python
describe_preset(preset: str) -> str
```

Get detailed description of a preset including all parameters.

**Example:**
```python
from kan_mlx_physics.presets import describe_preset

print(describe_preset("quantum_oscillator"))
# "quantum_oscillator: Quantum harmonic oscillator wavefunctions
#  - basis: hermite
#  - basis_kwargs: {'weighted': True}
#  - basis_M: 8"
```

---

## MultKAN Class

The main model class for building Kolmogorov-Arnold Networks.

**Location:** `kan_mlx_physics.multkan`

### Constructor

```python
MultKAN(
    width: Union[List[int], List[List[int]]],
    grid: int = 5,
    k: int = 3,
    noise_scale: float = 0.1,
    base_fun: Callable = nn.silu,
    grid_range: Tuple[float, float] = (-1.0, 1.0),
    seed: Optional[int] = None,
    mult_arity: int = 2,
    # Basis configuration
    basis: Union[str, List[str]] = "bspline",
    basis_M: Union[int, List[int], None] = None,
    basis_kwargs: Union[Dict, List[Dict], None] = None,
    # Descriptive aliases (these take precedence if provided)
    spline_order: Optional[int] = None,        # alias for k
    grid_points: Optional[int] = None,         # alias for grid
    multiplication_inputs: Optional[int] = None, # alias for mult_arity
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | `List[int]` or `List[List[int]]` | Required | Network architecture. Simple: `[2, 5, 1]`. Extended: `[[2, 0], [4, 1], [1, 0]]` for [n_sum, n_mult] |
| `grid` (`grid_points`) | `int` | 5 | Number of spline grid intervals |
| `k` (`spline_order`) | `int` | 3 | Spline order (0=step, 1=linear, 2=quadratic, 3=cubic) |
| `noise_scale` | `float` | 0.1 | Scale of random initialization for coefficients |
| `base_fun` | `Callable` | `nn.silu` | Base activation function (residual connection) |
| `grid_range` | `Tuple[float, float]` | `(-1.0, 1.0)` | Initial uniform grid range |
| `seed` | `Optional[int]` | None | Random seed for reproducibility |
| `mult_arity` (`multiplication_inputs`) | `int` | 2 | Number of inputs per multiplication node |
| `basis` | `str` or `List[str]` | `"bspline"` | Basis type(s): `"bspline"`, `"fourier"`, `"chebyshev"`, `"hermite"`, `"laguerre"`, `"legendre"`. Can be per-layer list. |
| `basis_M` | `int`, `List[int]`, or `None` | None | Number of basis functions. If None, uses `grid + k`. |
| `basis_kwargs` | `Dict`, `List[Dict]`, or `None` | None | Basis-specific parameters (see Basis Module section). |

**Parameter Aliases:**

For more readable code, you can use descriptive aliases:

| Original | Alias | Description |
|----------|-------|-------------|
| `k` | `spline_order` | Clearer meaning for spline polynomial order |
| `grid` | `grid_points` | Clearer meaning for number of grid intervals |
| `mult_arity` | `multiplication_inputs` | Describes what the parameter controls |

**Basis Examples:**

```python
# Uniform Fourier basis for all layers
model = MultKAN(width=[2, 10, 1], basis="fourier", basis_M=11)

# Per-layer basis configuration
model = MultKAN(
    width=[2, 10, 10, 1],
    basis=["chebyshev", "fourier", "bspline"],
    basis_M=[8, 15, 10],
)

# With basis-specific options
model = MultKAN(
    width=[1, 10, 1],
    basis="hermite",
    basis_kwargs={"weighted": True},  # Gaussian-weighted Hermite for QM
)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `width` | `List[int]` | Total nodes per layer |
| `depth` | `int` | Number of layers (excluding input) |
| `n_sum` | `List[int]` | Sum nodes per layer |
| `n_mult` | `List[int]` | Multiplication nodes per layer |
| `layers` | `List[KANLayer]` | List of layer modules |
| `symbolic_funs` | `List[Symbolic_KANLayer]` | Symbolic function containers |

### Forward Pass

```python
model(
    x: mx.array,
    return_activations: bool = False,
    singularity_avoiding: bool = False,
    y_th: float = 10.0,
) -> Union[mx.array, Tuple[mx.array, List[mx.array]]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `mx.array` | Required | Input tensor of shape `(batch, input_dim)` |
| `return_activations` | `bool` | False | Return intermediate activations |
| `singularity_avoiding` | `bool` | False | Clip large values to avoid singularities |
| `y_th` | `float` | 10.0 | Threshold for singularity avoidance |

**Returns:** Output tensor `(batch, output_dim)` or tuple with activations.

### Training Methods

#### fit()

```python
model.fit(
    dataset: Dict[str, mx.array],
    opt: str = "Adam",
    steps: int = 100,
    lr: float = 1e-2,
    batch_size: int = -1,
    lamb: float = 0.0,
    lamb_l1: float = 1.0,
    lamb_entropy: float = 2.0,
    lamb_coef: float = 0.0,
    lamb_coefdiff: float = 0.0,
    update_grid: bool = True,
    grid_update_freq: int = 100,
    stop_grid_update_step: int = -1,
    loss_fn: Optional[Callable] = None,
    log: int = 10,
    # Descriptive aliases (these take precedence if provided)
    learning_rate: Optional[float] = None,    # alias for lr
    regularization: Optional[float] = None,   # alias for lamb
    l1_weight: Optional[float] = None,        # alias for lamb_l1
    entropy_weight: Optional[float] = None,   # alias for lamb_entropy
    coef_weight: Optional[float] = None,      # alias for lamb_coef
    smooth_weight: Optional[float] = None,    # alias for lamb_coefdiff
) -> Dict[str, List[float]]
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `dataset` | Dict with `train_input`, `train_label`, optionally `test_input`, `test_label` |
| `opt` | Optimizer: `"Adam"`, `"AdamW"`, `"SGD"`, or `"LBFGS"` |
| `steps` | Number of training iterations |
| `lr` (`learning_rate`) | Learning rate |
| `batch_size` | Batch size (-1 for full batch) |
| `lamb` (`regularization`) | Overall regularization strength |
| `lamb_l1` (`l1_weight`) | L1 coefficient penalty weight |
| `lamb_entropy` (`entropy_weight`) | Entropy regularization (promotes sparsity) |
| `lamb_coef` (`coef_weight`) | Coefficient magnitude penalty |
| `lamb_coefdiff` (`smooth_weight`) | Coefficient smoothness penalty |
| `update_grid` | Whether to update grids during training |
| `grid_update_freq` | Steps between grid updates |
| `stop_grid_update_step` | Stop grid updates after this step (-1 = never) |
| `loss_fn` | Custom loss function (default: MSE) |
| `log` | Print frequency (0 = silent) |

**Parameter Aliases for fit():**

| Original | Alias | Description |
|----------|-------|-------------|
| `lr` | `learning_rate` | Standard ML terminology |
| `lamb` | `regularization` | Clearer meaning |
| `lamb_l1` | `l1_weight` | Clearer meaning |
| `lamb_entropy` | `entropy_weight` | Clearer meaning |
| `lamb_coef` | `coef_weight` | Clearer meaning |
| `lamb_coefdiff` | `smooth_weight` | Clearer meaning |

**Returns:** Dictionary with training history (`train_loss`, `test_loss`, `reg`).

#### forward_fast()

```python
model.forward_fast(x: mx.array) -> mx.array
```

JIT-compiled forward pass (no symbolic, faster for inference).

#### update_grid_from_samples()

```python
model.update_grid_from_samples(x: mx.array) -> None
```

Update spline grids to better fit data distribution.

### Architecture Modification

#### refine()

```python
model.refine(new_grid: int) -> None
```

Increase grid resolution while preserving learned functions.

#### prune()

```python
model.prune(
    threshold: float = 1e-2,
    mode: str = "auto",
    active_neurons_id: Optional[List] = None,
) -> None
```

Prune unimportant edges and nodes.

#### prune_edges()

```python
model.prune_edges(threshold: float = 0.01) -> None
```

Set low-importance edges to zero.

#### prune_nodes()

```python
model.prune_nodes(threshold: float = 0.01) -> None
```

Remove nodes with low contribution.

#### prune_input()

```python
model.prune_input(threshold: float = 1e-2) -> List[int]
```

Remove inactive input dimensions. Returns list of removed indices.

#### expand_width()

```python
model.expand_width(layer_idx: int, new_width: int) -> None
```

Add neurons to a layer.

#### expand_depth()

```python
model.expand_depth(new_layer_width: int, position: int = -1) -> None
```

Insert a new hidden layer.

### Symbolic Regression

#### fix_symbolic()

```python
model.fix_symbolic(
    l: int,
    i: int,
    j: int,
    fn_name: str,
    fit_params: bool = True,
    x: Optional[mx.array] = None,
    a_range: Tuple[float, float] = (-10, 10),
    b_range: Tuple[float, float] = (-10, 10),
) -> None
```

Fix an edge to a symbolic function.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `l` | Layer index |
| `i` | Input node index |
| `j` | Output node index |
| `fn_name` | Function name (e.g., `"sin"`, `"exp"`) |
| `fit_params` | Fit affine parameters (a, b, c, d) for `c*f(a*x+b)+d` |
| `x` | Sample data for parameter fitting |
| `a_range`, `b_range` | Search ranges for affine parameters |

#### unfix_symbolic()

```python
model.unfix_symbolic(l: int, i: int, j: int) -> None
```

Remove symbolic assignment from an edge.

#### suggest_symbolic()

```python
model.suggest_symbolic(
    l: int,
    i: int,
    j: int,
    x: Optional[mx.array] = None,
    top_k: int = 5,
    a_range: Tuple[float, float] = (-10, 10),
    b_range: Tuple[float, float] = (-10, 10),
) -> List[Tuple[str, float, float, float, float, float, float]]
```

Suggest best symbolic functions for an edge.

**Returns:** List of tuples: `(fn_name, r2, a, b, c, d, complexity_score)`

#### auto_symbolic()

```python
model.auto_symbolic(
    x: Optional[mx.array] = None,
    r2_threshold: float = 0.99,
    a_range: Tuple[float, float] = (-10, 10),
    b_range: Tuple[float, float] = (-10, 10),
) -> List[Tuple[int, int, int, str]]
```

Automatically detect and fix symbolic functions.

**Returns:** List of fixed edges as `(layer, i, j, fn_name)` tuples.

### Formula Output

#### symbolic_formula()

```python
model.symbolic_formula(
    var_names: Optional[List[str]] = None,
    decimals: int = 2,
    simplify: bool = False,
) -> str
```

Get formula in Unicode format.

#### symbolic_formula_latex()

```python
model.symbolic_formula_latex(
    var_names: Optional[List[str]] = None,
    decimals: int = 2,
) -> str
```

Get formula in LaTeX format.

#### symbolic_formula_typst()

```python
model.symbolic_formula_typst(
    var_names: Optional[List[str]] = None,
    decimals: int = 2,
) -> str
```

Get formula in Typst format.

#### to_sympy()

```python
model.to_sympy(var_names: Optional[List[str]] = None) -> sympy.Expr
```

Convert to SymPy expression.

#### to_tree()

```python
model.to_tree() -> Dict
```

Export as expression tree dictionary.

### Model Management

#### saveckpt() / loadckpt()

```python
model.saveckpt(path: str = "model") -> None
model.loadckpt(path: str = "model") -> None
```

Save/load model checkpoint (pickle format).

#### Versioning

```python
model._save_version(name: Optional[str] = None) -> None
model.rewind(version: int = -1) -> None
model.checkout(version: int) -> None
model.list_versions() -> List[str]
```

### Other Methods

#### predict_with_uncertainty()

```python
model.predict_with_uncertainty(
    x: mx.array,
    n_samples: int = 100,
    noise_scale: float = 0.01,
) -> Tuple[mx.array, mx.array]
```

Get predictions with uncertainty estimates (mean, std).

#### edge_scores()

```python
model.edge_scores(
    metric: str = "backward",
    x: Optional[mx.array] = None,
) -> List[mx.array]
```

Compute edge importance scores.

**Metrics:** `"backward"`, `"forward"`, `"magnitude"`, `"entropy"`, `"hybrid"`

#### summary()

```python
model.summary() -> None
```

Print architecture summary.

#### plot()

```python
model.plot(
    folder: str = "./figures",
    beta: float = 3.0,
    metric: str = "backward",
    scale: float = 0.5,
    tick: bool = False,
    sample: bool = False,
    in_vars: Optional[List[str]] = None,
    out_vars: Optional[List[str]] = None,
    title: Optional[str] = None,
    display: bool = True,
    save: bool = True,
) -> None
```

Generate PyKAN-style network visualization.

---

## KANLayer Class

Individual layer in a KAN network with pluggable basis functions.

**Location:** `kan_mlx_physics.kan_layer`

### Constructor

```python
KANLayer(
    in_dim: int,
    out_dim: int,
    num_grid: int = 5,
    k: int = 3,
    noise_scale: float = 0.1,
    base_fun: Callable = nn.silu,
    grid_range: Tuple[float, float] = (-1.0, 1.0),
    # Basis configuration
    basis: str = "bspline",
    basis_M: Optional[int] = None,
    basis_kwargs: Optional[Dict[str, Any]] = None,
)
```

**Basis Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `basis` | `str` | `"bspline"` | Basis type: `"bspline"`, `"fourier"`, `"chebyshev"`, `"hermite"`, `"laguerre"`, `"legendre"` |
| `basis_M` | `int` or `None` | None | Number of basis functions. If None, uses `num_grid + k`. |
| `basis_kwargs` | `Dict` or `None` | None | Basis-specific options (see table below) |

**Basis-Specific Options:**

| Basis | Option | Type | Default | Description |
|-------|--------|------|---------|-------------|
| `fourier` | `learnable_freq` | `bool` | `True` | Learn frequency parameter |
| `fourier` | `base_freq` | `float` | `1.0` | Base frequency |
| `hermite` | `weighted` | `bool` | `False` | Multiply by exp(-x²/2) for QM wavefunctions |
| `laguerre` | `alpha` | `float` | `0.0` | Generalized Laguerre parameter L_n^α |
| `chebyshev` | `learnable_affine` | `bool` | `True` | Learn input normalization |
| `legendre` | `learnable_affine` | `bool` | `True` | Learn input normalization |

### Learnable Parameters

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `coef` | `(in_dim, out_dim, M)` | Basis coefficients (M = num basis functions) |
| `scale_sp` | `(in_dim, out_dim)` | Spline output scale |
| `scale_base` | `(in_dim, out_dim)` | Base function scale |
| `basis_shift` | `(in_dim,)` | Input shift (non-bspline only) |
| `basis_scale_raw` | `(in_dim,)` | Input scale (non-bspline only) |
| `basis_omega_raw` | `(in_dim,)` | Frequency (Fourier only) |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `basis_type` | `str` | Current basis type |
| `num_basis_features` | `int` | Number of basis functions per edge |
| `symbolic_priority` | `List[str]` | Functions to prioritize in symbolic regression |

### Methods

```python
layer(x: mx.array, return_activations: bool = False) -> mx.array
layer.update_grid_from_samples(x: mx.array, margin: float = 0.01, grid_eps: float = 0.02) -> None
layer.set_mask(mask: mx.array) -> None
layer.get_subset(in_ids: List[int], out_ids: List[int]) -> KANLayer
layer.edge_scores() -> mx.array
```

---

## Basis Module

Pluggable basis functions for KAN edge parameterization.

**Location:** `kan_mlx_physics.basis`

### Registry Functions

```python
from kan_mlx_physics.basis import make_basis, list_bases, recommend_basis

# Create a basis by name
basis = make_basis("fourier", M=11, learnable_freq=True)

# List all available bases
print(list_bases())
# ['bspline', 'fourier', 'chebyshev', 'hermite', 'laguerre', 'legendre']

# Get recommendations for a physics problem
bases = recommend_basis("quantum_mechanics", "bounded")
# ['hermite', 'chebyshev', 'legendre']

bases = recommend_basis("periodic")
# ['fourier']
```

### Basis Types

#### FourierBasis

Fourier basis: `[1, cos(ωx), sin(ωx), cos(2ωx), sin(2ωx), ...]`

```python
from kan_mlx_physics.basis import FourierBasis, BasisConfig

basis = FourierBasis(
    config=BasisConfig(M=11),
    learnable_freq=True,   # Learn ω
    base_freq=1.0,         # Initial ω
)
```

**Best for:** Periodic functions, wave equations, Bloch wavefunctions.

#### ChebyshevBasis

Chebyshev polynomials T_n(x) on [-1, 1].

```python
from kan_mlx_physics.basis import ChebyshevBasis

basis = ChebyshevBasis(
    config=BasisConfig(M=10, domain=(-1, 1)),
    learnable_affine=True,  # Learn input normalization
)
```

**Best for:** Spectral methods, bounded intervals, high-accuracy approximation.

#### HermiteBasis

Physicist's Hermite polynomials H_n(x).

```python
from kan_mlx_physics.basis import HermiteBasis

basis = HermiteBasis(
    config=BasisConfig(M=8),
    weighted=True,  # Multiply by exp(-x²/2) for QM
)
```

**Best for:** Quantum harmonic oscillator, Gaussian-weighted problems.

#### LaguerreBasis

Generalized Laguerre polynomials L_n^α(x) on [0, ∞).

```python
from kan_mlx_physics.basis import LaguerreBasis

basis = LaguerreBasis(
    config=BasisConfig(M=8, domain=(0, float('inf'))),
    alpha=1.0,  # Generalized Laguerre L_n^1
)
```

**Best for:** Radial problems, hydrogen atom wavefunctions.

#### LegendreBasis

Legendre polynomials P_n(x) on [-1, 1].

```python
from kan_mlx_physics.basis import LegendreBasis

basis = LegendreBasis(
    config=BasisConfig(M=8, domain=(-1, 1)),
)
```

**Best for:** Angular momentum, spherical harmonics, Legendre ODEs.

#### BSplineBasis

B-spline basis (default, wraps legacy implementation).

```python
from kan_mlx_physics.basis import BSplineBasis

basis = BSplineBasis.from_grid(num_grid=5, k=3)
```

**Best for:** General function approximation, adaptive grids.

### Basis Abstract Class

All bases inherit from `Basis`:

```python
class Basis:
    name: str                    # Basis identifier
    num_features: int            # Number of basis functions (M)
    symbolic_priority: List[str] # Functions to prioritize in symbolic regression

    def init_params(self, in_dim: int, out_dim: Optional[int] = None) -> Dict[str, mx.array]
        """Initialize learnable parameters (shift, scale, omega)."""

    def features(self, x: mx.array, params: Dict[str, mx.array]) -> mx.array
        """Evaluate basis functions. Returns shape (batch, in_dim, M)."""
```

### Utility Functions

```python
from kan_mlx_physics.basis import contract_basis_coef, softplus

# Contract basis features with coefficients
# phis: (batch, in_dim, M), coef: (in_dim, out_dim, M) -> (batch, in_dim, out_dim)
output = contract_basis_coef(phis, coef)

# Softplus for positive parameterization
scale = softplus(scale_raw) + 1e-6
```

---

## Functional API

Pure functions for gradient computation and optimization.

**Location:** `kan_mlx_physics.functional`

### Type Definitions

```python
ParamsTuple = Tuple[mx.array, mx.array, mx.array, mx.array]  # (coef, scale_sp, scale_base, grid)
ParamsList = List[ParamsTuple]  # One tuple per layer
```

### Parameter Management

```python
get_params_list(model: MultKAN) -> ParamsList
```
Extract parameters from model.

```python
set_params_list(model: MultKAN, params_list: ParamsList) -> None
```
Update model with new parameters.

```python
flatten_params(params_list: ParamsList) -> Tuple[np.ndarray, Dict[str, Any]]
```
Flatten to 1D array for scipy optimizers.

```python
unflatten_params(flat: np.ndarray, metadata: Dict[str, Any]) -> ParamsList
```
Reconstruct from flattened array.

### Forward Pass

```python
kan_layer_forward(
    x: mx.array,
    coef: mx.array,
    scale_sp: mx.array,
    scale_base: mx.array,
    grid: mx.array,
    k: int,
    base_fun: Callable,
) -> mx.array
```
Pure forward pass for a single layer.

```python
functional_forward(
    params_list: ParamsList,
    x: mx.array,
    k: int,
    base_fun: Callable,
) -> mx.array
```
Pure forward pass for entire network. Works with `mx.grad()`.

### Optimization

```python
init_adam_state(params_list: ParamsList) -> Tuple[ParamsList, ParamsList, int]
```
Initialize Adam optimizer state (m, v, t).

```python
adam_update(
    params: ParamsList,
    grads: ParamsList,
    m: ParamsList,
    v: ParamsList,
    t: int,
    lr: float = 0.001,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[ParamsList, ParamsList, ParamsList, int]
```
Pure Adam update step.

```python
lbfgs_optimize(
    loss_fn: Callable[[ParamsList], mx.array],
    params_list: ParamsList,
    max_iter: int = 100,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    history_size: int = 10,
    callback: Optional[Callable] = None,
    verbose: bool = False,
) -> Tuple[ParamsList, scipy.optimize.OptimizeResult]
```
L-BFGS optimization using scipy.

```python
lbfgs_fit(
    model: MultKAN,
    dataset: Dict[str, mx.array],
    loss_fn: Optional[Callable] = None,
    max_iter: int = 100,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    verbose: bool = True,
) -> Tuple[MultKAN, scipy.optimize.OptimizeResult]
```
Convenience function to fit model with L-BFGS.

---

## PINN Utilities

Physics-Informed Neural Network utilities with batch-grad sum trick.

**Location:** `kan_mlx_physics.pinn`

### Derivative Functions

```python
make_u_fn(k: int, base_fun: Callable) -> Callable
```
Create function value evaluator: `u_fn(params, x) -> (N,)`

```python
make_derivative_fns(k: int, base_fun: Callable) -> Tuple[Callable, Callable, Callable]
```
Create u, du/dx, d2u/dx2 functions.

```python
make_laplacian_fn(k: int, base_fun: Callable, dim: int = 1) -> Callable
```
Create Laplacian function: `laplacian(params, x) -> (N,)`

```python
make_gradient_fn(k: int, base_fun: Callable) -> Callable
```
Create gradient function: `gradient(params, x) -> (N,) or (N, dim)`

### PINNOperators Class

```python
class PINNOperators:
    def __init__(self, k: int, base_fun: Callable, dim: int = 1)

    def u(self, params: ParamsList, x: mx.array) -> mx.array
    def gradient(self, params: ParamsList, x: mx.array) -> mx.array
    def laplacian(self, params: ParamsList, x: mx.array) -> mx.array
    def dx(self, params: ParamsList, x: mx.array, component: int = 0) -> mx.array
    def d2x(self, params: ParamsList, x: mx.array, component: int = 0) -> mx.array
```

### Compiled Training

```python
make_compiled_pinn_step(
    k: int,
    base_fun: Callable,
    loss_fn: Callable,
    lr: float = 0.003,
    dim: int = 1,
) -> Callable
```
Create a `@mx.compile` decorated training step.

### Finite Differences

```python
finite_difference_laplacian(
    forward_fn: Callable,
    x: mx.array,
    h: float = 1e-3,
) -> mx.array
```
Compute Laplacian using finite differences (for validation).

---

## Spline Functions

B-spline basis functions and utilities.

**Location:** `kan_mlx_physics.spline`

```python
extend_grid(grid: mx.array, k_extend: int = 0) -> mx.array
```
Add boundary points to grid.

```python
B_batch(x: mx.array, grid: mx.array, k: int = 0) -> mx.array
```
Evaluate B-spline basis functions using de Boor recursion.

```python
coef2curve(x: mx.array, grid: mx.array, coef: mx.array, k: int) -> mx.array
```
Evaluate spline from coefficients.

```python
curve2coef(x: mx.array, y: mx.array, grid: mx.array, k: int) -> mx.array
```
Fit spline coefficients from data (least squares).

---

## Symbolic Functions

Symbolic function registry and regression utilities.

**Location:** `kan_mlx_physics.symbolic`

### SymbolicFunction Dataclass

```python
@dataclass
class SymbolicFunction:
    name: str
    fn: Callable                    # MLX function
    fn_np: Callable                 # NumPy function
    latex: str                      # LaTeX representation
    typst: str = ""                 # Typst representation
    complexity: int = 1             # Complexity score (1-5)
    safe_fn: Optional[Callable]     # Singularity-safe version
    safe_fn_np: Optional[Callable]
    domain: str = "all"             # Valid domain ("all", "x >= 0", "x != 0", "|x| <= 1")
```

### Registry Functions

```python
add_symbolic(
    name: str,
    fn: Callable,
    fn_np: Callable = None,
    latex: str = None,
    typst: str = "",
    complexity: int = 1,
) -> None
```
Register a custom symbolic function.

```python
list_symbolic() -> List[str]
```
List all registered function names.

```python
SYMBOLIC_REGISTRY: Dict[str, SymbolicFunction]
```
The global registry (can be accessed directly).

### Fitting Functions

```python
fit_affine_params(
    x: np.ndarray,
    y: np.ndarray,
    fn_name: str,
    a_range: Tuple[float, float] = (-10, 10),
    b_range: Tuple[float, float] = (-10, 10),
    n_search: int = 50,
) -> Tuple[float, float, float, float, float]
```
Fit `y = c*f(a*x+b) + d`. Returns `(a, b, c, d, r2)`.

```python
suggest_symbolic(
    x: np.ndarray,
    y: np.ndarray,
    a_range: Tuple[float, float] = (-10, 10),
    b_range: Tuple[float, float] = (-10, 10),
    n_search: int = 50,
    weight_simple: float = 0.1,
    top_k: int = 3,
) -> List[Tuple[str, float, float, float, float, float, float]]
```
Suggest best functions. Returns list of `(name, r2, a, b, c, d, score)`.

```python
validate_domain(fn_name: str, x: np.ndarray) -> bool
```
Check if x values are in function's valid domain.

```python
score_symbolic_fit(r2: float, complexity: int, weight_simple: float = 0.1) -> float
```
Compute weighted score (accuracy - complexity penalty).

### Built-in Functions

| Category | Functions |
|----------|-----------|
| Identity | `x`, `0`, `1` |
| Powers | `x^2`, `x^3`, `x^4`, `x^0.5`, `x^-1`, `x^-2` |
| Trigonometric | `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan` |
| Exponential | `exp`, `log` |
| Hyperbolic | `sinh`, `cosh`, `tanh` |
| Other | `abs`, `sign`, `gaussian`, `sigmoid`, `relu`, `softplus` |

### Physics Symbolic Functions

**Location:** `kan_mlx_physics.physics_symbolic`

```python
register_physics_symbolic() -> None
```
Register all physics functions (call once).

```python
list_physics_symbolic() -> Dict[str, List[str]]
```
List physics functions by category.

**Categories:**
- Quantum Mechanics: Hermite, Laguerre, radial wavefunctions, Legendre, Chebyshev
- Special Functions: Bessel, Airy, Gamma, Error functions
- QFT: Propagator, Bose-Einstein, Fermi-Dirac, Planck
- Cosmology: Scale factor evolution, Schwarzschild
- Deformation Quantization: q-exponential, Moyal basis functions

---

## Visualization

Plotting utilities for KAN networks.

**Location:** `kan_mlx_physics.visualization`

```python
plot_kan(
    model: MultKAN,
    folder: str = "./figures",
    beta: float = 3.0,
    metric: str = "backward",
    scale: float = 0.5,
    tick: bool = False,
    sample: bool = False,
    in_vars: Optional[List[str]] = None,
    out_vars: Optional[List[str]] = None,
    title: Optional[str] = None,
    display: bool = True,
    save: bool = True,
) -> None
```
PyKAN-identical network visualization.

```python
plot_activations(
    model: MultKAN,
    layer_idx: int,
    x: Optional[mx.array] = None,
    folder: str = "./figures",
    display: bool = True,
    save: bool = True,
) -> None
```
Plot all activation functions in a layer.

```python
plot_training_history(
    history: Dict[str, List[float]],
    metric: str = "loss",
    display: bool = True,
    save: bool = True,
    folder: str = "./figures",
) -> None
```
Plot training curves.

```python
plot_spline_1d(
    model: MultKAN,
    layer_idx: int,
    x_sample: Optional[mx.array] = None,
    display: bool = True,
) -> None
```
Plot 1D spline curves.

```python
plot = plot_kan  # Alias
```

---

## Utilities

Dataset creation and test functions.

**Location:** `kan_mlx_physics.utils`

### Dataset Creation

```python
create_dataset(
    f: Callable,
    n_var: int = 2,
    ranges: Union[float, Tuple[float, float], List[Tuple[float, float]]] = (-1, 1),
    train_num: int = 1000,
    test_num: int = 1000,
    normalize_input: bool = False,
    normalize_label: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, mx.array]
```
Create dataset from function.

**Returns:** Dict with keys `train_input`, `train_label`, `test_input`, `test_label`.

```python
create_dataset_from_data(
    inputs: mx.array,
    labels: mx.array,
    train_ratio: float = 0.8,
    normalize_input: bool = False,
    normalize_label: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, mx.array]
```
Create dataset from existing data.

### Test Functions

```python
f_sin(x: mx.array) -> mx.array       # sin(pi * sum(x))
f_exp(x: mx.array) -> mx.array       # exp(sum(x))
f_gaussian(x: mx.array) -> mx.array  # exp(-sum(x^2))
f_polynomial(x: mx.array) -> mx.array # x^2 + y^2
f_mixed(x: mx.array) -> mx.array     # sin(pi*x) + y^2
```

---

## PDE Module

High-level PDE solving with domain-specific language.

**Location:** `kan_mlx_physics.pde`

### Main Solver

```python
solve(
    equation: str,
    domain: Union[List, Dict] = [-1, 1],
    params: Dict = {},
    solver_config: Optional[SolverConfig] = None,
) -> Tuple[MultKAN, Dict]
```
Solve PDE from string specification.

**Example:**
```python
psi, history = solve(
    "-nabla^2 psi/2 + x^2 psi/2 = E psi",
    domain=[-5, 5],
    params={"E": 0.5}
)
```

### SolverConfig

```python
@dataclass
class SolverConfig:
    width: List[int] = [1, 20, 20, 1]
    grid: int = 10
    k: int = 3
    steps: int = 1000
    lr: float = 0.01
    optimizer: str = "Adam"
    n_interior: int = 1000
    n_boundary: int = 200
    sampling: str = "sobol"  # "uniform", "sobol", "adaptive"
    lambda_pde: float = 1.0
    lambda_bc: float = 10.0
    lambda_reg: float = 0.001
```

### Differential Operators

**Location:** `kan_mlx_physics.pde.operators`

```python
grad(f: Callable, x: mx.array, h: float = 1e-4) -> mx.array
laplacian(f: Callable, x: mx.array, h: float = 1e-4) -> mx.array
div(F: Callable, x: mx.array, h: float = 1e-4) -> mx.array
curl(F: Callable, x: mx.array, h: float = 1e-4) -> mx.array  # 3D only
```

### Moyal Star Product

```python
moyal_bracket(
    f: Callable,
    g: Callable,
    x: mx.array,
    hbar: float = 1.0,
    h: float = 1e-4,
) -> mx.array
```
Compute Moyal bracket {f, g}_M.

```python
star_product(
    f: Callable,
    g: Callable,
    x: mx.array,
    hbar: float = 1.0,
    order: int = 2,
    h: float = 1e-4,
) -> mx.array
```
Compute Moyal star product f *_theta g.

### Pre-built Physics Problems

**Location:** `mlx_kan.pde.physics`

```python
class Schrodinger:
    """1D Schrodinger equation"""

class WheelerDeWitt:
    """Wheeler-DeWitt quantum cosmology"""

class DeformedSchrodinger:
    """Schrodinger with Moyal deformation"""

class DeformedWheelerDeWitt:
    """Wheeler-DeWitt with Moyal deformation"""

class KleinGordonCurved:
    """Klein-Gordon on curved spacetime"""

class Friedmann:
    """Friedmann cosmology equations"""
```

---

## Formula Rendering

Multi-format formula output.

**Location:** `kan_mlx_physics.formula_render`

```python
class FormulaRenderer:
    def render_formula(expr) -> str      # Unicode
    def render_latex(expr) -> str        # LaTeX

@dataclass
class FormulaTerm:
    name: str
    coefficient: float

def render_formula(model: MultKAN, var_names: List[str], format: str = "text") -> str
def print_formula_box(formula: str) -> None  # Pretty-print in ASCII box
```

---

## Constants and Type Hints

```python
from kan_mlx_physics.functional import ParamsTuple, ParamsList
from kan_mlx_physics.symbolic import SYMBOLIC_REGISTRY, SymbolicFunction
```

---

## Error Handling

All functions raise standard Python exceptions:
- `ValueError` for invalid parameters
- `RuntimeError` for computation failures
- `FileNotFoundError` for missing checkpoints

Training gracefully handles NaN/Inf by logging warnings and continuing.
