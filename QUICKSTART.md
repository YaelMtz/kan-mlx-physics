# Quickstart

KAN-MLX-Physics runs on **Apple Silicon** (MLX). These steps get you from a fresh
clone to a solved physics PDE in a couple of minutes.

## Install

With [`uv`](https://docs.astral.sh/uv/) (recommended — reproducible from the
committed lockfile):

```bash
git clone https://github.com/yaelmartinez/kan-mlx-physics.git
cd kan-mlx-physics
uv sync                      # creates the environment from uv.lock
```

Or with plain `pip`:

```bash
pip install git+https://github.com/yaelmartinez/kan-mlx-physics.git
```

## Run a physics example

The flagship example discovers the harmonic-oscillator **Wigner function** from
its phase-space PDE alone — no analytic answer supplied:

```bash
uv run python examples/wigner_harmonic_oscillator.py
```

Expected output (near-exact recovery of `E = 1/2` and `W_0 = e^{-r^2}/pi`;
exact figures vary slightly with seed and hardware):

```
Result (no analytic target was used in training):
  eigenvalue   E = 0.5014   (exact 0.5,  error 0.27%)
  peak      W(0) = 0.3107   (exact 0.3183)
  purity int W^2 = 0.04828  (exact 0.05066)
  L2 vs analytic = 2.4%
```

## Minimal API

Fit a 1-D function and read back a symbolic formula:

```python
import mlx.core as mx
from kan_mlx_physics import KAN, create_dataset

data = create_dataset(lambda x: mx.sin(3.14159 * x[:, 0]), n_var=1, ranges=(-1, 1))
model = KAN(width=[1, 3, 1], grid=5, k=3)
model.fit(data, steps=200)
print(model.symbolic_formula())     # -> ~ sin(3.14 x)
```

## Solve a PDE with the physics DSL

```python
from kan_mlx_physics.pde import PDEBuilder, PDEResidualLoss, AnchorLoss

# 1-D Poisson: u'' = -pi^2 sin(pi x),  u(0)=u(1)=0
model, hist = (
    PDEBuilder("Derivative(u, x, 2) = -pi**2 * sin(pi*x)")
    .domain([0, 1])
    .loss(PDEResidualLoss(weight=1.0))
    .loss(AnchorLoss(x0=mx.array([[0.0]]), target=0.0, weight=100.0, id="l"))
    .loss(AnchorLoss(x0=mx.array([[1.0]]), target=0.0, weight=100.0, id="r"))
    .phase("train", steps=2000, lr=1e-3)
    .model(width=[1, 8, 1], basis="fourier")
    .solve()
)
```

## Reproduce the paper's experiments

Every primary recovery experiment is data-free (no analytic target in training).
Given `uv sync`, run any example with `uv run python examples/<name>.py`. See the
`benchmarks/` directory for timing scripts and environment notes; report the Apple
chip, MLX version, architecture, collocation count, derivative order, and precision
when quoting performance.

## Notes & limits

- **Apple Silicon required** (MLX). See the README *Scope & limitations*.
- **Second-order-derivative PINN losses are slow** (nested autodiff is not
  `mx.compile`-able); expect ~20-40× the per-step cost of first-order losses.
