# Benchmark: kan_mlx_physics (MLX) vs pykan (PyTorch)

Run it yourself:

```bash
uv run python3 kan-mlx-physics/benchmark_kan_vs_pykan.py --try-mps
```

## Why the old numbers were wrong

The previous root-level `comparison_results.json` / `full_comparison.log` (now
suffixed `.INVALID`) reported "pykan is ~20× faster" (MLX 0.58s vs pykan 0.03s).
**Those numbers are invalid.** Both runs had `"success": false` — they crashed on
`ModuleNotFoundError` (`torch` / `mlx` not importable in the spawned subprocess),
so the "times" are just interpreter-startup-until-import-crash, not training.
`compare_kans.py` also timed whole subprocesses (startup + import + JIT warmup),
never steady state.

## How this benchmark is fair

- **In-process** — no subprocess startup inside the timer.
- **Identical setup** — same architecture, data, #steps, LR, optimizer (Adam),
  and a hand-written `forward → MSE → backward → step` loop on both sides.
- **Warmup excluded** — the first steps pay import/JIT/graph-compile costs
  (MLX lazy-compiles; PyTorch warms caches). We run warmup steps untimed, then
  time steady-state steps only, syncing the device before start/stop.
- **Speed *and* accuracy** — final MSE reported for both (speed is meaningless
  if the fits differ in quality).
- **Honest device labels** — MLX runs on the Apple-Silicon GPU (Metal); pykan on
  CPU (torch's stable path on this Mac). This is the framework-choice comparison
  an Apple-Silicon user faces, not a same-device kernel comparison.

## Representative result (task: fit sin(πx), Adam, 200 timed steps)

With `mx.compile` on the MLX step (the default for first-order fits — see the
"compilation" note below), **MLX wins across the board**, and the lead widens
sharply with model size while pykan-CPU throughput collapses:

| Model | MLX (compiled) | pykan (CPU) | Winner |
|-------|---------------:|------------:|--------|
| `[1,5,1]`, grid 5   | 1857 s/s | 1006 s/s | **MLX 1.85×** |
| `[1,20,1]`, grid 10 | 1818 s/s |  570 s/s | **MLX 3.2×** |
| `[1,50,1]`, grid 10 | 1694 s/s |  167 s/s | **MLX 10.2×** |
| `[1,100,1]`, grid 10| 1830 s/s |  143 s/s | **MLX 12.8×** |

Accuracy is identical with or without compilation (MLX MSE ≈ 3e-5, pykan ≈ 1e-4
on the baseline). pykan-MPS was *slower* than pykan-CPU for these small models,
confirming CPU is the right pykan baseline here.

**Before compilation** there was a crossover: uncompiled MLX ran ~360–620 s/s
and *lost* to pykan-CPU on tiny models (GPU dispatch overhead dominates), only
winning ~3× on large ones. `mx.compile` fuses the Metal kernels and removes that
per-step dispatch overhead — a 4–5× raw step speedup — which erases the
crossover entirely.

**Takeaway:** with compilation, MLX is the right choice at every size, and by a
large margin for the medium/large models this library targets.

## A note on compilation and second-order PDEs

`mx.compile` is applied automatically in `MultKAN.fit` for first-order fitting.
It is **not** applied to second-order PDE residuals (nested `mx.grad`, as in the
Wigner / Schrödinger workloads): MLX currently raises `[Compiled] Cannot vjp
primitive` when a compiled step contains grad-of-grad composed with a VJP. Those
training loops run uncompiled (still on the GPU). Grid-update steps also skip
compilation because they change the graph structure.

Numbers are machine-dependent — regenerate on your hardware with the command
above. Raw JSON is written to `benchmark_kan_vs_pykan_results.json`.
