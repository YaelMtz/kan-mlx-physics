# Changelog

All notable changes to KAN-MLX-Physics are documented here. This project follows
[Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/) format.

## [Unreleased]

### Added
- **`PDEBuilder.param_bounds(**bounds)` — allow non-positive trainable eigenvalues.** The
  trainer clamped every trainable parameter to `≥ 0.01` after each step (a positivity
  projection), which is right for an energy above a well bottom but WRONG for problems
  whose eigenvalue is genuinely negative — the Morse oscillator, hydrogen, and the
  Lorentzian Wheeler–DeWitt / cosmological constraints. `param_bounds(E=(None, None))`
  lets such an eigenvalue train freely; unlisted params keep the historical `≥ 0.01`
  default, so existing problems are unchanged. Plumbed through `pde(..., param_bounds=…)`.
- **`TransitionLoss(left_eq, right_eq, …)` — coupled left/right ⋆-genvalue residual.** A
  non-diagonal (transition) Wigner distribution `W_{E_L E_R}` must satisfy `H⋆W = E_L W`
  AND `W⋆H = E_R W` simultaneously with two independent eigenvalues. This loss compiles
  both equation strings and sums their squared residuals, each reading its own trainable
  eigenvalue; pair it with `trainable_params("E_L","E_R")`. Enables the non-diagonal
  anisotropic-cosmology (Taub / Kantowski–Sachs) problems.

### Fixed
- **Mixed partial derivatives are now evaluated correctly.** `Derivative(w, x, t)` (and
  any multi-coordinate derivative) silently collapsed to the first-axis derivative only
  (`∂_x` instead of `∂_x∂_t`), because the compiler took `wrt[0]`. Mixed-derivative nodes
  are now detected in the residual tree and computed via nested autodiff (`∂²u/∂x_i∂x_j`),
  verified against the analytic `∂²[sin x cos t]/∂x∂t`. This matters for the transformed
  (Mellin-real) phase-space equations, whose kinetic cross-term is a genuine mixed partial.
- **Declared coordinates are now authoritative over the `KNOWN_COORDS` guess list.**
  A parameter whose name happened to be in `KNOWN_COORDS` (e.g. the Morse width `a`,
  or `r`) was miscast as a *coordinate* even when `.coordinates(...)` was used,
  corrupting the coordinate→column map and crashing the residual with a shape
  mismatch. With explicit coordinates, only the declared names are coordinates.
- **Per-coordinate derivatives are reliably reduced to their own column.** A second
  derivative with respect to a non-first coordinate (e.g. `Derivative(w, t, 2)`)
  could leak an `(N, dim)` array into the residual and fail to broadcast against the
  scalar-per-point terms (`(N)` vs `(N, dim)`); a defensive column reduction now
  guarantees the right shape. Together these let the full Mellin-transformed
  phase-space equations (mixed `∂_x∂_t`, second `∂_t²`, exponential potentials)
  build and solve.

### Added
- **`PDEBuilder.coordinates(*names)` — declare the independent variables explicitly.**
  The parser recognizes a fixed set of coordinate names (`x, y, z, t, r, r2, …`); any
  other bare symbol appearing only algebraically — notably a momentum (`p`, `px`, `py`)
  in a phase-space / Wigner equation — was otherwise silently classified as a
  *parameter*, left without a value, and crashed at evaluation (`mx.full(..., None)`).
  Declaring coordinates removes the ambiguity and makes genuine phase-space PDEs
  first-class:
  ```python
  (PDEBuilder("(p**2/2 - E)*W - (hbar**2/8)*Derivative(W, x, 2) = 0")
       .coordinates("x", "p")          # p is a coordinate, not a parameter
       .params(hbar=1.0, E=0.5).trainable_params("E")
       .domain([[0, L], [-P, P]]))     # bounds follow the declared coordinate order
  ```
  Also plumbed through the functional `pde(..., coords=[...])` entry point and
  `parser.parse(..., extra_coords=[...])`.
- **Dimension inference is now coordinate-driven.** The spatial/phase-space dimension
  is the number of non-time coordinates actually present, so declared momenta count
  correctly (a `(x, p)` Wigner problem reports `dimension == 2`); it no longer depends
  on a hardcoded spatial-name set.
- **`viz.paper.network(model, ...)`** — a single, publication-styled KAN network
  figure (pykan-style: the learned activation is drawn on each edge; symbolic edges
  render red). Wraps `MultKAN.plot` in the chainable `Figure` surface
  (`.caption(...).save(...)`).
- **`viz.paper.symbolic_refinement(model, x, ...)`** — the interpretability story in
  one declarative call: a side-by-side *pre-symbolic* (learned spline edges, black)
  vs *post-symbolic* (edges fixed to named physics functions via `auto_symbolic`,
  red) network comparison, with the recovered formula placed beneath. The returned
  `Figure` exposes `.formula`.

### Fixed
- `viz.paper`'s `Figure.caption(...)` now reserves bottom margin (`reserve=`) so the
  caption sits below the x-axis label instead of overprinting it.

## [0.3.0] - 2026-09-14

### Added
- **`solve_spectrum(...)` — eigenstate discovery as a flagship workflow.** Sequential,
  deflation-based recovery of a whole spectrum from a star-genvalue / eigenvalue PDE
  with no supervised target: each excited state is found while enforcing
  Hilbert–Schmidt orthogonality (`Tr(ρₙρₘ)=0`) to the already-recovered lower states.
  Returns a `Spectrum` with `.eigenvalues`, `.states`, `.residuals`, `.overlaps()`
  (Gram matrix) and `.table()`. Ships a graph-stable `DeflationLoss` (frozen priors
  pre-tabulated on a fixed grid).
- **`compare_priors(...)` + `Prior` — functional inductive priors as a first-class
  concept.** Run the *same* physics problem under several named priors (basis +
  config) across seeds and get the selection/accuracy table (`P_select`, L2|correct)
  that a basis-prior study needs — the methodology of cross-representation
  inductive-bias research, in one call. Resumable via `Sweep`.
- **Operator-identity loss terms promoted to the library:** `TraceLoss` (`Tr ρ = 1`)
  and `PurityLoss` (`Tr ρ² = 1`), matching the operator-identity supervision described
  below; previously these lived only in experiment scripts.
- `kan_mlx_physics.viz` — two declarative visualization surfaces for two jobs:
  - **`viz.paper`** — publication figures: `paper.use()` applies a colorblind-safe
    (Okabe–Ito), serif/Computer-Modern, vector-PDF style; `paper.vs_analytic(...)`,
    `paper.wigner_2d(...)`, and `paper.convergence(...)` each render a common physics
    figure in one chained call (`.caption(...).save("fig.pdf")`). The Wigner surface
    uses a **diverging colormap centered at zero** — a Wigner function is a
    quasi-probability, so its sign is physical and must not be hidden by a sequential
    or rainbow map.
  - **`viz.live`** — a real-time `rich` terminal dashboard for experimenting:
    `live.monitor(analytic=W, total_steps=N)` returns a *zero-weight* loss hook that
    renders per-term loss, the trainable eigenvalue, and a live blind L2 vs an
    analytic reference while the solve runs. Headless/SSH-safe; degrades to a
    one-line print if `rich` is unavailable.

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
