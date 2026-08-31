# Toy-Problem Comparison — fit f(x) = sin(πx)

A rigorous, matched head-to-head on a well-known problem with a well-known
analytic answer (sin(πx) — pykan's canonical demo), across four KAN approaches.

Run:
```bash
uv run python3 kan-mlx-physics/toy_comparison.py        # runs the 4-way benchmark
uv run python3 kan-mlx-physics/toy_comparison_plot.py   # renders toy_comparison.png
```

## Setup (identical across all contenders)

- **Problem:** fit `f(x) = sin(πx)` on `[-1, 1]`, 200 training points.
- **Architecture:** `[1, 1, 1]`, grid 5, spline order k=3.
- **Optimizer:** Adam, lr 0.01, 1200 steps (warmup excluded from timing).
- **Loss:** MSE. Same seed.

## Contenders

1. **MLX B-spline** — `kan_mlx_physics`, uncompiled.
2. **MLX B-spline (compiled)** — same, with `mx.compile`.
3. **MLX Fourier** — `kan_mlx_physics` with the Fourier basis (sin(πx) *is* a
   Fourier mode, so this is the "physics-appropriate base function").
4. **pykan B-spline** — the PyTorch reference implementation, CPU.

## Results

| Contender | L2 error | steps/sec | wall (s) |
|-----------|---------:|----------:|---------:|
| MLX B-spline            | **0.35%** | 458  | 2.62 |
| MLX B-spline (compiled) | **0.35%** | **1774** | **0.68** |
| MLX Fourier (compiled)  | 1.06%     | 1397 | 0.86 |
| pykan B-spline          | 0.53%     | 915  | 1.31 |

(Machine-dependent — regenerate on your hardware. MLX runs on the Apple-Silicon
GPU; pykan on CPU.)

### Findings

- **Accuracy:** MLX B-spline (0.35%) is the *most accurate*, beating pykan
  (0.53%). Compiled vs uncompiled is **identical** — `mx.compile` is lossless.
- **Speed:** MLX-compiled is **3.9× faster than uncompiled MLX** and **1.9×
  faster than pykan**. Uncompiled MLX (458) actually *loses* to pykan (915) on
  this tiny model — confirming that on small models the win comes entirely from
  compilation removing GPU dispatch overhead.
- **Basis choice:** here the plain B-spline edged out Fourier on both accuracy
  and final loss. Fourier's symbolic form still recovers a clean `sin`, but its
  learnable-frequency affine wrapping made the numeric fit slightly looser for
  this single-mode target. (For multi-mode or periodic targets the Fourier basis
  is expected to pull ahead.)
- **Symbolic recovery:** all recover a `sin(...)` formula. pykan's is the
  cleanest (`sin(1.8·sin(2.04·x+…))`); the MLX forms carry more affine nesting.

## Convergence, accuracy, speed, and fit

See `toy_comparison.png` — four panels: overlaid convergence curves, accuracy
bars, speed bars, and fit-vs-analytic (all four overlay sin(πx) near-perfectly).

## Takeaway

On the canonical KAN toy problem, `kan_mlx_physics` **matches or beats pykan on
accuracy and, once compiled, is ~2× faster** — a fair, reproducible result that
replaces the earlier invalid `comparison_results.json` (which came from crashed
imports). The right basis matters, but for a single sine mode the general
B-spline is already excellent.
