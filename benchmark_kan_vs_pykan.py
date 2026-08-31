#!/usr/bin/env python3
"""Honest, in-process benchmark: kan_mlx_physics (MLX) vs pykan (PyTorch).

Replaces the old comparison_results.json / full_comparison.log, whose numbers
came from two CRASHED subprocess runs (ModuleNotFoundError: torch / mlx) and
therefore timed interpreter-startup-until-import-crash, not training.

What this does differently — and why it is fair:
  * Both frameworks run IN THIS PROCESS (no subprocess startup in the timer).
  * IDENTICAL setup: same architecture, same data, same #steps, same LR, same
    optimizer (Adam), same manual loop (forward -> MSE -> backward -> step).
  * WARMUP before timing: the first steps pay import/JIT/graph-compile costs
    (MLX lazy-compiles; PyTorch warms caches). We run warmup steps untimed,
    then time steady-state steps only.
  * Reports BOTH speed (steps/sec) AND accuracy (final MSE) — speed is
    meaningless if the two fit to different quality.
  * A scaling sweep over grid size and network width.

Device note (reported, not hidden): MLX runs on the Apple-Silicon GPU (Metal);
pykan/PyTorch runs on CPU by default here (its most stable path). That is the
honest, relevant comparison for an Apple-Silicon user choosing a framework —
not a same-device apples-to-apples of the math kernels. Both are labeled in the
output. A pykan-MPS run is attempted too when available.

Run:  uv run python3 kan-mlx-physics/benchmark_kan_vs_pykan.py
"""

import argparse
import json
import time

import numpy as np


# ----------------------------------------------------------------------------
# Timing helper: warm up, then time steady-state steps.
# ----------------------------------------------------------------------------
def timed_steps(step_fn, sync_fn, warmup, measured):
    """Run `warmup` untimed steps, then time `measured` steps. Returns
    (seconds_per_step, last_loss)."""
    last = None
    for _ in range(warmup):
        last = step_fn()
    sync_fn()  # make sure warmup is fully materialized before we start timing
    t0 = time.perf_counter()
    for _ in range(measured):
        last = step_fn()
    sync_fn()  # materialize all timed work before stopping the clock
    dt = time.perf_counter() - t0
    return dt / measured, float(last)


# ----------------------------------------------------------------------------
# MLX (kan_mlx_physics)
# ----------------------------------------------------------------------------
def run_mlx(width, grid, k, n_points, lr, warmup, measured, seed=0, compile=True):
    from functools import partial

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from kan_mlx_physics import MultKAN

    x = mx.linspace(-1, 1, n_points).reshape(-1, 1)
    y = mx.sin(3.14159265 * x)

    model = MultKAN(width=width, grid=grid, k=k, seed=seed, grid_range=(-1.0, 1.0))
    opt = optim.Adam(learning_rate=lr)

    def loss_fn(m):
        return mx.mean((m(x) - y) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    if compile:
        # Compile the fused loss+grad+update step. The model and optimizer
        # state are captured as compile inputs/outputs so the graph is reused
        # across steps — this fuses Metal kernels and removes per-step dispatch
        # overhead, the dominant cost for small models on the GPU.
        state = [model.state, opt.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def step():
            loss, grads = loss_and_grad(model)
            opt.update(model, grads)
            return loss
    else:
        def step():
            loss, grads = loss_and_grad(model)
            opt.update(model, grads)
            return loss

    def sync():
        mx.eval(model.parameters(), opt.state)

    sps, final_loss = timed_steps(step, sync, warmup, measured)
    tag = "MLX (Metal GPU, compiled)" if compile else "MLX (Metal GPU)"
    return {"backend": tag, "sec_per_step": sps,
            "steps_per_sec": 1.0 / sps, "final_mse": final_loss}


# ----------------------------------------------------------------------------
# pykan (PyTorch)
# ----------------------------------------------------------------------------
def run_pykan(width, grid, k, n_points, lr, warmup, measured, seed=0, device="cpu"):
    import torch
    from kan import KAN

    torch.manual_seed(seed)
    x = torch.linspace(-1, 1, n_points, device=device).reshape(-1, 1)
    y = torch.sin(3.14159265 * x)

    model = KAN(width=list(width), grid=grid, k=k, seed=seed, device=device,
                symbolic_enabled=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def step():
        opt.zero_grad()
        out = model(x)
        loss = torch.mean((out - y) ** 2)
        loss.backward()
        opt.step()
        return loss.item()

    def sync():
        if device == "mps":
            torch.mps.synchronize()

    sps, final_loss = timed_steps(step, sync, warmup, measured)
    return {"backend": f"pykan (PyTorch {device.upper()})", "sec_per_step": sps,
            "steps_per_sec": 1.0 / sps, "final_mse": final_loss}


def _fmt(r):
    return (f"{r['backend']:26} {r['steps_per_sec']:8.1f} steps/s "
            f"{r['sec_per_step']*1e3:8.2f} ms/step   MSE={r['final_mse']:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measured", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--try-mps", action="store_true",
                    help="Also benchmark pykan on Apple MPS (may be unstable).")
    ap.add_argument("--out", default="benchmark_kan_vs_pykan_results.json")
    args = ap.parse_args()

    print("=" * 78)
    print("HONEST BENCHMARK — kan_mlx_physics (MLX) vs pykan (PyTorch)")
    print(f"Task: fit sin(pi*x) on [-1,1].  warmup={args.warmup} timed={args.measured} "
          f"steps, Adam lr={args.lr}")
    print("Steady-state timing (warmup excluded); identical arch/data/steps.")
    print("=" * 78)

    all_results = {}

    # --- 1. Matched baseline: [1,5,1], grid=5 ---
    base = dict(width=[1, 5, 1], grid=5, k=3, n_points=100, lr=args.lr,
                warmup=args.warmup, measured=args.measured)
    print("\n[1] Baseline  width=[1,5,1] grid=5 k=3 n=100")
    r_mlx = run_mlx(**base)
    r_pk = run_pykan(**base, device="cpu")
    print("   ", _fmt(r_mlx))
    print("   ", _fmt(r_pk))
    print(f"    -> MLX is {r_pk['sec_per_step']/r_mlx['sec_per_step']:.2f}x "
          f"pykan-CPU throughput")
    all_results["baseline"] = {"mlx": r_mlx, "pykan_cpu": r_pk}
    if args.try_mps:
        try:
            r_mps = run_pykan(**base, device="mps")
            print("   ", _fmt(r_mps))
            all_results["baseline"]["pykan_mps"] = r_mps
        except Exception as e:
            print(f"    (pykan-MPS failed: {type(e).__name__}: {e})")

    # --- 2. Scaling over grid size ---
    print("\n[2] Scaling over grid  (width=[1,5,1], grid in {5,10,20,50})")
    grid_scan = []
    for g in [5, 10, 20, 50]:
        cfg = dict(width=[1, 5, 1], grid=g, k=3, n_points=100, lr=args.lr,
                   warmup=args.warmup, measured=args.measured)
        rm = run_mlx(**cfg)
        rp = run_pykan(**cfg, device="cpu")
        ratio = rp["sec_per_step"] / rm["sec_per_step"]
        print(f"    grid={g:3}  MLX {rm['steps_per_sec']:7.1f} s/s | "
              f"pykan-CPU {rp['steps_per_sec']:7.1f} s/s | MLX {ratio:.2f}x")
        grid_scan.append({"grid": g, "mlx": rm, "pykan_cpu": rp, "ratio": ratio})
    all_results["grid_scaling"] = grid_scan

    # --- 3. Scaling over width ---
    print("\n[3] Scaling over hidden width  ([1,W,1], grid=10, W in {5,20,50,100})")
    width_scan = []
    for w in [5, 20, 50, 100]:
        cfg = dict(width=[1, w, 1], grid=10, k=3, n_points=100, lr=args.lr,
                   warmup=args.warmup, measured=args.measured)
        rm = run_mlx(**cfg)
        rp = run_pykan(**cfg, device="cpu")
        ratio = rp["sec_per_step"] / rm["sec_per_step"]
        print(f"    W={w:3}  MLX {rm['steps_per_sec']:7.1f} s/s | "
              f"pykan-CPU {rp['steps_per_sec']:7.1f} s/s | MLX {ratio:.2f}x")
        width_scan.append({"width": w, "mlx": rm, "pykan_cpu": rp, "ratio": ratio})
    all_results["width_scaling"] = width_scan

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {args.out}")
    print("\nNOTE: MLX=Metal GPU, pykan=CPU (torch's stable path on this Mac).")
    print("This is the framework-choice comparison for an Apple-Silicon user,")
    print("not a same-device kernel comparison. Accuracy (MSE) reported for both.")


if __name__ == "__main__":
    main()
