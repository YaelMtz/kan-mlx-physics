# Changelog

All notable changes to KAN-MLX-Physics are documented here. This project follows
[Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/) format.

## [Unreleased]

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
