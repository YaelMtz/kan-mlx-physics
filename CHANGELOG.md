# Changelog

All notable changes to KAN-MLX-Physics are documented here. This project follows
[Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/) format.

## [Unreleased]

## [0.2.0] - 2026-09-14

### Added
- `Sweep` — a minimal, resumable multi-seed experiment helper. A full parameter ×
  seed grid with resume-on-restart, JSONL logging, and a grouped median±std summary
  now takes ~5 lines instead of ~60 of hand-written boilerplate:
  `Sweep(path, **axes)` → `@sweep.run` cell function → `sweep.summary(group_by, show)`.
- Operator-identity supervision: a family of data-free physics losses whose target
  values are fixed by universal operator identities rather than by an (unknown)
  analytic solution — pure-state purity (`Tr ρ² = 1`), vanishing quantum energy
  variance (`⟨H²⟩ − ⟨H⟩² = 0`, via the exact Moyal relation for polynomial `H`), and
  Hilbert–Schmidt deflation (`Tr(ρₙρₘ) = 0`) for progressive excited-state recovery.
  These compose with the trace and PDE-residual terms to enable eigenstate discovery
  in phase space with no supervised target.
- Wigner–Laguerre–Gaussian symbolic primitives `WL_0`…`WL_4` (`e^{-s} L_n(2s)`),
  enabling native-family symbolic identification of harmonic-oscillator Wigner
  eigenfunctions.
- Performance and scope documentation (README): unified-memory rationale,
  second-order-autodiff cost, honest reproducibility boundaries.

### Fixed
- **`predict_with_uncertainty` no longer corrupts model weights.** It drew fresh
  random noise to "remove" the perturbation instead of subtracting the noise it had
  added, leaving a net random walk that permanently damaged the weights over the
  sampling loop. It now records and undoes the exact perturbation.
- **`MultKAN(basis=None)` no longer raises.** `None` now normalizes to the default
  B-spline (matching `KANLayer`); previously `len(basis)` raised `TypeError`.
- **Unknown functions in a PDE equation now raise instead of silently defaulting.**
  An unregistered `V(x)` was quietly replaced by the harmonic-oscillator potential
  `0.5*x**2` (and `U(a)` by `a**3`), so a forgotten `.function(...)` trained against
  the wrong physics. Register the function, or get a clear error.
- **Residual-based adaptive sampling (RBAS) is now actually applied.** `TrainingPhase`'s
  `use_rbas`/`rbas_weight`/`rbas_oversample` were never passed to the sampler and
  silently no-op'd; they are now threaded through.
- `use_gradnorm` now emits a clear warning that GradNorm balancing is not yet wired
  into the loss composer (previously it was silently ignored).
- Mixed-basis-slot (`basis_per_mult_slot`) layers now fail loudly on pruning
  (`get_subset`) and warn on grid update, instead of silently dropping their per-slot
  parameters and corrupting the model.
- Removed a duplicate `state` property on `FunctionalOptimizer` that shadowed the
  correct flat-tensor version with a list-of-tuples one (broke `mx.eval(opt.state)`).
- Checkpointing now persists the full model configuration (basis type,
  `basis_kwargs`, multiplication-node structure, `noise_scale`) and the trainer's
  trainable parameters (e.g. the eigenvalue `E`); previously a non-B-spline model
  round-tripped as a default B-spline.
- General-basis (Hermite/Laguerre) training step fused into a single
  `value_and_grad` pass.
- Cross-library reference tests are robust to PyKAN API changes
  (`.layers` → `.act_fun`); they now locate the grid attribute or skip cleanly.
- Test suite is warning-free (converted `return`-style tests to `assert`).
- Packaging metadata modernized: SPDX `license = "MIT"`, `license-files`,
  `[dependency-groups]`.

### Known limitations
- Second-order-autodiff PINN losses are not `mx.compile`-able (MLX limitation),
  bounding per-step throughput on the hardest operation.
- Pruning runs but does not yet reliably reduce width under the default threshold;
  aggressive L1 is required to collapse redundant edges.

## [0.1.0]

### Added
- Initial release: MLX-native Kolmogorov–Arnold Networks with a physics-first
  PDE DSL (`PDEBuilder`), trainable eigenvalues, orthogonal-polynomial and spline
  bases, Moyal ⋆-product support, symbolic extraction, and PyKAN-style
  visualization.
