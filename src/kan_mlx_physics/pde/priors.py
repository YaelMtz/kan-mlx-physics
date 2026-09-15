"""
Functional inductive priors and their comparison — the software embodiment of
cross-representation basis-prior research.

A *prior* is a named hypothesis space (basis + configuration) carrying a functional
inductive bias for a physical operator. `compare_priors` runs the *same* physics-
informed problem under several priors across seeds and reports the selection /
accuracy table (P_select, eigenvalue error, L2) that a basis-prior study needs —
turning the methodology into one call::

    from kan_mlx_physics.pde import compare_priors, Prior

    table = compare_priors(
        equation="...", domain=[0, 10], seeds=range(20),
        make_losses=lambda: [...],
        priors={
            "Hermite":  Prior(basis="weighted_hermite"),
            "Laguerre": Prior(basis="laguerre",
                              basis_kwargs={"alpha":0, "weighted":True,
                                            "nonnegative_input":True, "fixed_scale":2.0}),
            "B-spline": Prior(basis=None),
        },
        score=my_score_fn,            # returns {"nodes":.., "l2":.., "E":..}
    )
    print(table)                      # P_select / L2|correct per prior
"""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Prior:
    """A named functional inductive prior: a basis + its configuration.

    Attributes
    ----------
    basis:
        Basis name (``"weighted_hermite"``, ``"laguerre"``, ``"chebyshev"``, …) or
        ``None`` for the default B-spline.
    basis_kwargs:
        Basis-specific configuration.
    width, grid, k:
        Architecture / resolution for this prior's model.
    noise_scale:
        Coefficient-init noise.
    """
    basis: Optional[str] = None
    basis_kwargs: Dict[str, Any] = field(default_factory=dict)
    width: Sequence[int] = (1, 1)
    grid: int = 20
    k: int = 3
    noise_scale: float = 0.08

    def model_kwargs(self, **overrides) -> Dict[str, Any]:
        import mlx.core as mx
        mk = dict(width=list(self.width), grid=self.grid, k=self.k,
                  base_fun=lambda z: mx.zeros_like(z), noise_scale=self.noise_scale)
        if self.basis is not None:
            mk["basis"] = self.basis
        if self.basis_kwargs:
            mk["basis_kwargs"] = dict(self.basis_kwargs)
        mk.update(overrides)
        return mk


def compare_priors(
    equation: str,
    domain,
    priors: Dict[str, Prior],
    *,
    make_losses: Callable[[], Sequence[Any]],
    score: Callable[[Any, float], Dict[str, Any]],
    seeds: Sequence[int] = range(8),
    params: Optional[dict] = None,
    trainable: Sequence[str] = ("E",),
    E_init: float = 0.5,
    phases: Optional[Sequence[dict]] = None,
    correct: Optional[Callable[[Dict[str, Any]], bool]] = None,
    results_path: Optional[str] = None,
    grid_range=None,
    verbose: bool = True,
) -> "PriorComparison":
    """Run one physics problem under several priors × seeds; return the summary.

    Parameters
    ----------
    priors:
        ``{name: Prior(...)}`` — the hypothesis spaces to compare.
    make_losses:
        Zero-arg factory of the (fresh) loss terms for one run.
    score:
        ``score(model, E) -> {"nodes":.., "l2":.., ...}`` — your per-run metrics.
        Whatever keys it returns become columns.
    correct:
        Predicate on a score dict deciding whether the run "selected correctly"
        (drives ``P_select``). Default: a ``"correct"`` key if present, else the
        run is counted as correct when ``score`` did not set ``fail``.
    results_path:
        Optional JSONL for resumable runs (uses :class:`~kan_mlx_physics.Sweep`).

    Returns
    -------
    PriorComparison
        ``.table()`` / ``str()`` gives the P_select / median-L2 table; ``.rows``
        holds the raw per-run records.
    """
    from .dsl import PDEBuilder

    phases = phases or [dict(steps=6000, lr=1.5e-3, n_points=2500),
                        dict(steps=4000, lr=6e-4, n_points=2500)]
    if correct is None:
        def correct(s):  # noqa
            return (s.get("correct") if "correct" in s else not s.get("fail", False))

    lo, hi = domain
    gr = grid_range or (lo, hi)
    rows = []

    def run_one(name: str, seed: int):
        prior = priors[name]
        b = PDEBuilder(equation)
        p = dict(params or {}); p.setdefault("E", E_init)
        b = b.params(**p).trainable_params(*trainable).domain([lo, hi])
        for term in make_losses():
            b = b.loss(term)
        for ph in phases:
            b = b.phase(**{"name": ph.get("name", "p"), **ph})
        mk = prior.model_kwargs(seed=seed, grid_range=gr)
        model, hist = b.model(**mk).solve(verbose=False)
        E = float(np.array(hist.trainable_params.get("E", float("nan"))))
        sc = score(model, E)
        return {"prior": name, "seed": int(seed), "E": E, **sc}

    if results_path:  # resumable via Sweep
        from ..experiment import Sweep
        sw = Sweep(results_path, prior=list(priors.keys()), seed=list(seeds))

        @sw.run
        def cell(prior, seed):  # noqa
            r = run_one(prior, seed)
            r.pop("prior", None); r.pop("seed", None)
            return r
        rows = sw.results()
    else:
        for name in priors:
            for seed in seeds:
                r = run_one(name, seed)
                rows.append(r)
                if verbose:
                    print(f"  {name:12} seed={seed}: " +
                          " ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in r.items() if k not in ("prior", "seed")),
                          flush=True)

    return PriorComparison(list(priors.keys()), rows, correct)


class PriorComparison:
    """Summary of a :func:`compare_priors` run — the P_select / accuracy table."""

    def __init__(self, prior_names, rows, correct):
        self.prior_names = list(prior_names)
        self.rows = [r for r in rows if r is not None]
        self._correct = correct

    def summary(self):
        """Per-prior aggregates: P_select and median L2 | correct."""
        out = {}
        for name in self.prior_names:
            cs = [r for r in self.rows if r.get("prior", name) == name]
            if not cs:
                out[name] = dict(n=0); continue
            ok = [r for r in cs if self._correct(r)]
            l2ok = [r["l2"] for r in ok if "l2" in r]
            out[name] = dict(
                p_select=len(ok) / len(cs),
                l2_correct=(float(np.median(l2ok)) if l2ok else float("nan")),
                n=len(cs),
            )
        return out

    def table(self) -> str:
        s = self.summary()
        lines = [f"{'prior':>12}  {'P_select':>9}  {'L2|correct':>11}  {'n':>3}"]
        lines.append("  " + "-" * 40)
        for name in self.prior_names:
            d = s[name]
            if d.get("n", 0) == 0:
                lines.append(f"{name:>12}  {'(none)':>9}"); continue
            lines.append(f"{name:>12}  {d['p_select']:>9.2f}  "
                         f"{d['l2_correct']:>10.1f}%  {d['n']:>3}")
        return "\n".join(lines)

    def __str__(self):
        return self.table()

    def __repr__(self):
        return f"<PriorComparison: {len(self.prior_names)} priors, {len(self.rows)} runs>"


__all__ = ["Prior", "compare_priors", "PriorComparison"]
