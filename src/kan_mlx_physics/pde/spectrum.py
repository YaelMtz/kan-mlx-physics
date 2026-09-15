"""
Eigenstate discovery — the flagship spectral workflow.

Sequential, deflation-based recovery of a whole spectrum from a star-genvalue /
eigenvalue PDE, with no supervised target. Each excited state is found by solving
the physics-informed problem while enforcing Hilbert–Schmidt orthogonality to the
already-recovered lower states:

    L_defl^(n) = sum_{m<n} ( c * <psi_n, psi_m> )^2       (Tr(rho_n rho_m) = 0).

This wraps the recipe validated in the harmonic-oscillator study (recovery through
n=4 at P_select = 1.0) into one call::

    from kan_mlx_physics.pde import solve_spectrum

    spectrum = solve_spectrum(
        equation="(r2 - 2*E)*W - (hbar**2/4)*(4*r2*Derivative(W,r2,2)+4*Derivative(W,r2)) = 0",
        domain=[0, 10], n_states=5,
        make_losses=lambda: [PDEResidualLoss(50), PurityLoss(400), ...],
        model_kwargs=dict(width=[1,1], basis="laguerre", basis_kwargs={...}),
    )
    print(spectrum)                 # table: state | E | residual | overlap
    spectrum.eigenvalues            # [0.5, 1.5, 2.5, ...]
    spectrum.states                 # [model_0, model_1, ...]
    spectrum.overlaps()             # Gram matrix <psi_i, psi_j>

The DeflationLoss below is graph-stable (frozen priors are pre-tabulated on a fixed
grid, never re-run through the prior network each step) — the pattern that made
multi-state recovery practical.
"""
from __future__ import annotations
from typing import Any, Callable, List, Optional, Sequence
import math

import mlx.core as mx
import numpy as np

from .losses import LossTerm


class DeflationLoss(LossTerm):
    """Hilbert–Schmidt orthogonality to frozen recovered states.

    ``(scale * mean(W * W_m) * L)^2`` summed over priors, evaluated on a FIXED grid
    with the priors pre-tabulated as constants (`stop_gradient`). Data-free: the
    priors are recovered states, never an analytic target.
    """

    def __init__(self, priors: Sequence[Callable], domain,
                 weight: float = 5000.0, ngrid: int = 3000,
                 scale: float = 1.0, **kw):
        super().__init__(weight, **kw)
        lo, hi = domain
        self._xg = mx.linspace(float(lo), float(hi), ngrid).reshape(-1, 1)
        self._L = float(hi) - float(lo)
        self._scale = scale
        self._tab = []
        for pm in priors:
            wv = mx.stop_gradient(pm(self._xg))
            wv = mx.squeeze(wv, -1) if len(wv.shape) > 1 else wv
            self._tab.append(wv)

    def compute(self, ctx):
        if not self._tab:
            return mx.array(0.0)
        w = ctx.get_u(self._xg)
        w = mx.squeeze(w, -1) if len(w.shape) > 1 else w
        tot = mx.array(0.0)
        for wv in self._tab:
            overlap = mx.mean(w * wv) * self._L
            tot = tot + (self._scale * overlap) ** 2
        return tot


class Spectrum:
    """Result of :func:`solve_spectrum` — the recovered eigenstates and diagnostics."""

    def __init__(self, states, eigenvalues, residuals, domain, extras=None):
        self.states = list(states)
        self.eigenvalues = list(eigenvalues)
        self.residuals = list(residuals)
        self.domain = domain
        self.extras = extras or [{} for _ in states]

    def __len__(self):
        return len(self.states)

    def _grid(self, n=1000):
        lo, hi = self.domain
        return mx.linspace(float(lo), float(hi), n).reshape(-1, 1)

    def overlaps(self, n: int = 1000) -> "np.ndarray":
        """Gram matrix G[i,j] = <psi_i, psi_j> (mean * domain length) over the states.

        Off-diagonals near zero indicate successful orthogonalization.
        """
        xg = self._grid(n)
        L = float(self.domain[1]) - float(self.domain[0])
        cols = []
        for m in self.states:
            v = np.array(m(xg)).ravel()
            cols.append(v)
        k = len(cols)
        G = np.empty((k, k))
        for i in range(k):
            for j in range(k):
                G[i, j] = float(np.mean(cols[i] * cols[j]) * L)
        return G

    def table(self) -> str:
        """A compact text table of the spectrum."""
        lines = [f"{'state':>5}  {'eigenvalue':>12}  {'residual':>11}  {'extra':>0}"]
        lines.append("  " + "-" * 42)
        for n, (E, r) in enumerate(zip(self.eigenvalues, self.residuals)):
            extra = self.extras[n]
            xs = "  ".join(f"{k}={v:.3g}" for k, v in extra.items()
                          if isinstance(v, (int, float)))
            lines.append(f"{n:>5}  {E:>12.5f}  {r:>11.2e}  {xs}")
        return "\n".join(lines)

    def __repr__(self):
        return f"<Spectrum: {len(self)} states, E={[round(e,4) for e in self.eigenvalues]}>"

    def __str__(self):
        return self.table()


def solve_spectrum(
    equation: str,
    domain,
    n_states: int,
    *,
    make_losses: Callable[[], Sequence[LossTerm]],
    model_kwargs: dict,
    params: Optional[dict] = None,
    trainable: Sequence[str] = ("E",),
    phases: Optional[Sequence[dict]] = None,
    deflation_weight: float = 5000.0,
    deflation_scale: float = 1.0,
    residual_fn: Optional[Callable] = None,
    E_of: Callable[[int], float] = lambda n: n + 0.5,
    seed_of: Callable[[int], int] = lambda n: 3 + n,
    verbose: bool = True,
) -> Spectrum:
    """Recover ``n_states`` eigenstates sequentially with Hilbert–Schmidt deflation.

    Parameters
    ----------
    equation, domain, params, trainable, phases, model_kwargs:
        The per-state physics-informed problem, forwarded to :class:`PDEBuilder`.
        ``phases`` is a list of dicts passed verbatim to ``.phase(**d)``.
    make_losses:
        A zero-argument factory returning the (fresh) base loss terms for one state
        — e.g. ``lambda: [PDEResidualLoss(50), PurityLoss(400), DecayLoss(50), ...]``.
        A fresh list is required each state so loss objects are not shared.
    E_of, seed_of:
        Per-state eigenvalue initialization and RNG seed (default ``n + 1/2`` and
        ``3 + n``).
    deflation_weight, deflation_scale:
        The Hilbert–Schmidt deflation term added against all previously recovered
        states. ``scale`` is the ``c`` in ``(c<psi_n,psi_m>)^2`` (e.g. ``2*pi**2``
        for the Wigner normalization).

    Returns
    -------
    Spectrum
        ``.states``, ``.eigenvalues``, ``.residuals``, ``.overlaps()``, ``.table()``.
    """
    from .dsl import PDEBuilder

    phases = phases or [dict(steps=6000, lr=1.5e-3, n_points=3000),
                        dict(steps=4000, lr=6e-4, n_points=3000)]
    priors: List[Callable] = []
    states, eigenvalues, residuals, extras = [], [], [], []

    for n in range(n_states):
        b = PDEBuilder(equation)
        p = dict(params or {})
        p.setdefault("E", E_of(n))
        b = b.params(**p).trainable_params(*trainable).domain(list(domain))
        for term in make_losses():
            b = b.loss(term)
        if priors:
            b = b.loss(DeflationLoss(list(priors), domain,
                                     weight=deflation_weight, scale=deflation_scale))
        for ph in phases:
            b = b.phase(**{"name": ph.get("name", f"p{len(states)}"), **ph})
        mk = dict(model_kwargs)
        mk.setdefault("seed", seed_of(n))
        model, hist = b.model(**mk).solve(verbose=False)

        E = float(np.array(hist.trainable_params.get("E", float("nan"))))
        # residual proxy: the final total loss recorded by the trainer.
        res = float(getattr(hist, "final_loss", float("nan")))
        if getattr(hist, "losses", None):  # PhaseHistory-style, if present
            res = float(hist.losses[-1])
        if residual_fn is not None:
            res = float(residual_fn(model, n, E))
        states.append(model); eigenvalues.append(E); residuals.append(res)
        extras.append({})
        priors.append(model)
        if verbose:
            print(f"  state {n}: E={E:.5f}  residual={res:.2e}", flush=True)

    return Spectrum(states, eigenvalues, residuals, domain, extras)


__all__ = ["solve_spectrum", "Spectrum", "DeflationLoss"]
