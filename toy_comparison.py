#!/usr/bin/env python3
"""
Toy-problem comparison: KAN approaches on fit f(x) = sin(πx).

A rigorous, matched head-to-head on a well-known problem with a well-known
analytic answer (sin(πx)) — pykan's canonical demo. Four contenders:

  1. MLX  B-spline           (kan_mlx_physics, uncompiled)
  2. MLX  B-spline compiled  (kan_mlx_physics, mx.compile)
  3. MLX  Fourier basis      (kan_mlx_physics — the physics-appropriate basis:
                              sin(πx) IS a Fourier mode, so this should win on
                              accuracy AND symbolic recovery)
  4. pykan B-spline          (PyTorch, CPU — the reference implementation)

Everything matched: architecture [1,1,1], same 200 training points, same
Adam lr, same #steps, same loss (MSE). Warmup excluded from timing.

Measures:
  - Accuracy: L2 error of the fit vs analytic sin(πx)
  - Speed: steady-state steps/sec and wall time
  - Convergence: loss vs step (saved for overlay plot)
  - Symbolic: does each recover a sin(...) formula?

    uv run python3 kan-mlx-physics/toy_comparison.py
"""

import argparse
import json
import time

import numpy as np

PI = np.pi
STEPS = 1200
LR = 0.01
N = 200
WIDTH = [1, 1, 1]
GRID = 5
K = 3


def analytic(x):
    return np.sin(PI * x)


def l2_error(y_pred, x):
    y_true = analytic(x)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)) /
                 (np.sqrt(np.mean(y_true ** 2)) + 1e-12))


# ----------------------------------------------------------------------------
# MLX contenders
# ----------------------------------------------------------------------------
def run_mlx(basis, compile_step, warmup=20, measured=STEPS):
    from functools import partial
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from kan_mlx_physics import MultKAN, register_physics_symbolic
    register_physics_symbolic()

    xs = np.linspace(-1, 1, N).astype(np.float32)
    x = mx.array(xs.reshape(-1, 1))
    y = mx.array(analytic(xs).reshape(-1, 1).astype(np.float32))

    kwargs = dict(width=WIDTH, grid=GRID, k=K, seed=0, grid_range=(-1.0, 1.0))
    if basis == "fourier":
        kwargs["basis"] = "fourier"
    model = MultKAN(**kwargs)
    opt = optim.Adam(learning_rate=LR)

    def loss_fn(m, bx, by):
        return mx.mean((m(bx) - by) ** 2)

    lg = nn.value_and_grad(model, loss_fn)

    if compile_step:
        state = [model.state, opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step(bx, by):
            loss, grads = lg(model, bx, by)
            opt.update(model, grads)
            return loss
    else:
        def step(bx, by):
            loss, grads = lg(model, bx, by)
            opt.update(model, grads)
            return loss

    losses = []
    for _ in range(warmup):
        step(x, y)
    mx.eval(model.parameters(), opt.state)
    t0 = time.perf_counter()
    for i in range(measured):
        loss = step(x, y)
        if i % 10 == 0:
            losses.append(float(loss))
    mx.eval(model.parameters(), opt.state)
    wall = time.perf_counter() - t0

    y_pred = np.array(model(x)).flatten()
    l2 = l2_error(y_pred, xs)

    # symbolic recovery
    sym = "n/a"
    try:
        xsamp = mx.linspace(-1, 1, 200).reshape(-1, 1)
        fns = {"sin", "cos", "x"} if basis == "fourier" else {"sin", "cos", "x", "x^2"}
        model.auto_symbolic(x=xsamp, r2_threshold=0.0, allowed_fns=fns, verbose=False)
        sym = str(model.symbolic_formula(var_names=["x"], decimals=3, verbose=False))[:70]
    except Exception as e:
        sym = f"(symbolic failed: {type(e).__name__})"

    label = {"bspline": "MLX B-spline", "fourier": "MLX Fourier"}[basis]
    if compile_step:
        label += " (compiled)"
    return {"label": label, "backend": "MLX", "l2": l2, "wall": wall,
            "steps_per_sec": measured / wall, "losses": losses,
            "y_pred": y_pred.tolist(), "symbolic": sym}


# ----------------------------------------------------------------------------
# pykan contender
# ----------------------------------------------------------------------------
def run_pykan(warmup=20, measured=STEPS):
    import torch
    from kan import KAN

    torch.manual_seed(0)
    xs = np.linspace(-1, 1, N).astype(np.float32)
    x = torch.tensor(xs.reshape(-1, 1))
    y = torch.tensor(analytic(xs).reshape(-1, 1).astype(np.float32))

    model = KAN(width=WIDTH, grid=GRID, k=K, seed=0, device="cpu",
                symbolic_enabled=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    def step():
        opt.zero_grad()
        loss = torch.mean((model(x) - y) ** 2)
        loss.backward()
        opt.step()
        return loss.item()

    losses = []
    for _ in range(warmup):
        step()
    t0 = time.perf_counter()
    for i in range(measured):
        loss = step()
        if i % 10 == 0:
            losses.append(loss)
    wall = time.perf_counter() - t0

    with torch.no_grad():
        y_pred = model(x).numpy().flatten()
    l2 = l2_error(y_pred, xs)

    sym = "n/a"
    try:
        model.auto_symbolic(lib=["sin", "x", "x^2"], verbose=0)
        model(x)
        sym = str(model.symbolic_formula()[0][0])[:70]
    except Exception as e:
        sym = f"(symbolic failed: {type(e).__name__})"

    return {"label": "pykan B-spline", "backend": "pykan(CPU)", "l2": l2,
            "wall": wall, "steps_per_sec": measured / wall, "losses": losses,
            "y_pred": y_pred.tolist(), "symbolic": sym}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="toy_comparison_results.json")
    args = ap.parse_args()

    print("=" * 74)
    print(f"TOY COMPARISON — fit f(x)=sin(πx),  arch {WIDTH}, grid {GRID}, "
          f"{STEPS} steps, Adam lr={LR}")
    print("=" * 74)

    results = []
    print("\nRunning MLX B-spline (uncompiled)…")
    results.append(run_mlx("bspline", compile_step=False))
    print("Running MLX B-spline (compiled)…")
    results.append(run_mlx("bspline", compile_step=True))
    print("Running MLX Fourier…")
    results.append(run_mlx("fourier", compile_step=True))
    print("Running pykan B-spline (CPU)…")
    results.append(run_pykan())

    # Table
    print(f"\n{'contender':24} {'L2 err':>10} {'steps/s':>9} {'wall(s)':>8}  symbolic")
    print("-" * 90)
    for r in results:
        print(f"{r['label']:24} {r['l2']*100:>9.3f}% {r['steps_per_sec']:>9.0f} "
              f"{r['wall']:>8.2f}  {r['symbolic']}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {args.out}")


if __name__ == "__main__":
    main()
