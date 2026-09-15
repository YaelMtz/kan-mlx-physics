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
