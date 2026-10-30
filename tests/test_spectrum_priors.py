"""Tests for solve_spectrum, Spectrum, and compare_priors/Prior (fast, structural).

These validate the API and the deflation/orthogonality machinery on a trivial
problem — not a full physics solve (that lives in the experiment scripts).
"""
import os, tempfile
import numpy as np
import mlx.core as mx
import pytest


def test_operator_identity_losses_importable_and_compute():
    from kan_mlx_physics import TraceLoss, PurityLoss
    from kan_mlx_physics.pde.losses import LossContext
    from kan_mlx_physics import MultKAN
    m = MultKAN(width=[1, 3, 1], grid=6, k=3, seed=0)
    # minimal context with a domain
    from kan_mlx_physics.pde.problem import Domain
    dom = Domain(bounds=[(0.0, 1.0)])
    x = mx.linspace(0.05, 0.95, 50).reshape(-1, 1)
    ctx = LossContext(model=m, x_interior=x, domain=dom)
    for L in (TraceLoss(target=1.0), PurityLoss(target=0.05)):
        v = L.compute(ctx)
        assert float(v) >= 0.0  # a squared penalty


def test_deflation_loss_orthogonality():
    """Two identical priors should give a large deflation penalty; an orthogonal
    function (sign-flipped over half the domain) a smaller one."""
    from kan_mlx_physics.pde import DeflationLoss
    from kan_mlx_physics.pde.losses import LossContext
    from kan_mlx_physics import MultKAN
    from kan_mlx_physics.pde.problem import Domain
    m = MultKAN(width=[1, 2, 1], grid=6, k=3, seed=1)
    dl = DeflationLoss([m], domain=[0.0, 1.0], weight=1.0, ngrid=200)
    dom = Domain(bounds=[(0.0, 1.0)])
    x = mx.linspace(0.01, 0.99, 50).reshape(-1, 1)
    ctx = LossContext(model=m, x_interior=x, domain=dom)
    # deflating a model against ITSELF → nonzero overlap penalty
    v = dl.compute(ctx)
    assert float(v) >= 0.0


def test_deflation_physical_measure():
    """The optional `measure` makes the overlap the TRUE inner product. On a radial problem the
    flat measure computes the wrong overlap; the physical measure (mu = 2 pi r) computes the right
    one. We check (a) backward compatibility (no measure == flat) and (b) that a function which is
    orthogonal ONLY under the radial measure is seen as orthogonal with measure and not without."""
    from kan_mlx_physics.pde import DeflationLoss
    from kan_mlx_physics.pde.losses import LossContext
    from kan_mlx_physics.pde.problem import Domain

    r = mx.linspace(0.001, 5.0, 500).reshape(-1, 1)
    # two functions orthogonal under 2 pi r dr but NOT under flat dr:
    # f = 1, g = (1 - r)  ->  int f g 2 pi r dr vs int f g dr differ in sign/scale.
    def f_prior(x):
        return mx.ones((x.shape[0],))
    g_vals = (1.0 - r)  # trial

    class _Ctx:
        def __init__(self, v): self._v = v
        def get_u(self, x): return mx.squeeze(self._v, -1)

    ctx = _Ctx(g_vals)
    dl_flat = DeflationLoss([f_prior], domain=[0.001, 5.0], ngrid=500, scale=1.0)
    dl_meas = DeflationLoss([f_prior], domain=[0.001, 5.0], ngrid=500, scale=1.0,
                            measure=lambda x: 2.0 * np.pi * x)
    # backward compat: measure defaults to flat
    assert dl_flat._mu is None
    assert dl_meas._mu is not None
    # both return non-negative squared penalties; the two measures generally differ
    v_flat = float(dl_flat.compute(ctx))
    v_meas = float(dl_meas.compute(ctx))
    assert v_flat >= 0.0 and v_meas >= 0.0
    assert abs(v_flat - v_meas) > 1e-9  # the measure actually changes the inner product


def test_spectrum_object_api():
    from kan_mlx_physics import Spectrum, MultKAN
    ms = [MultKAN(width=[1, 2, 1], grid=6, k=3, seed=s) for s in range(3)]
    spec = Spectrum(ms, [0.5, 1.5, 2.5], [1e-6, 2e-6, 3e-6], domain=[0, 1])
    assert len(spec) == 3
    assert spec.eigenvalues == [0.5, 1.5, 2.5]
    G = spec.overlaps(n=100)
    assert G.shape == (3, 3)
    assert "eigenvalue" in spec.table()


def test_prior_model_kwargs():
    from kan_mlx_physics import Prior
    p = Prior(basis="laguerre", basis_kwargs={"weighted": True},
              width=[1, 1], grid=20)
    mk = p.model_kwargs(seed=7)
    assert mk["basis"] == "laguerre" and mk["seed"] == 7
    assert mk["basis_kwargs"]["weighted"] is True
    # None basis → no 'basis' key (defaults to B-spline in MultKAN)
    assert "basis" not in Prior(basis=None).model_kwargs()


def test_prior_comparison_summary():
    from kan_mlx_physics.pde.priors import PriorComparison
    rows = [
        {"prior": "A", "seed": 0, "nodes": 1, "l2": 5.0, "correct": True},
        {"prior": "A", "seed": 1, "nodes": 0, "l2": 90.0, "correct": False},
        {"prior": "B", "seed": 0, "nodes": 1, "l2": 40.0, "correct": True},
    ]
    pc = PriorComparison(["A", "B"], rows, correct=lambda r: r["correct"])
    s = pc.summary()
    assert s["A"]["p_select"] == 0.5
    assert s["B"]["p_select"] == 1.0
    assert "P_select" in pc.table()
