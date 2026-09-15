"""Tests for the two viz surfaces (paper + live)."""
import os, tempfile
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless


def _tmp(ext="pdf"):
    fd, p = tempfile.mkstemp(suffix="." + ext); os.close(fd); return p


def test_paper_use_and_palette():
    from kan_mlx_physics.viz import paper
    paper.use()
    assert "analytic" in paper.PALETTE and len(paper.CATEGORICAL) >= 4


def test_paper_vs_analytic_saves():
    from kan_mlx_physics.viz import paper
    paper.use()
    r = np.linspace(0, 6, 50); Wt = np.exp(-r); Wk = Wt * 1.01
    p = _tmp()
    paper.vs_analytic(r, Wk, Wt).save(p)
    assert os.path.getsize(p) > 0; os.remove(p)


def test_paper_wigner_2d_diverging_saves():
    from kan_mlx_physics.viz import paper
    paper.use()
    x = np.linspace(-3, 3, 40); X, P = np.meshgrid(x, x)
    W = np.exp(-(X**2 + P**2)) * (1 - 2*(X**2 + P**2))  # has negatives
    assert (W < 0).any()
    p = _tmp()
    paper.wigner_2d(X, P, W).save(p)
    assert os.path.getsize(p) > 0; os.remove(p)


def test_paper_convergence_from_dict():
    from kan_mlx_physics.viz import paper
    paper.use()
    p = _tmp()
    paper.convergence({"losses": list(np.exp(-np.arange(100)/20))}).save(p)
    assert os.path.getsize(p) > 0; os.remove(p)


def test_live_monitor_is_zero_weight_hook():
    from kan_mlx_physics.viz import live
    mon = live.monitor(total_steps=100, every=10, title="t")
    # weight 0 so it never changes the objective
    assert getattr(mon, "weight", 0.0) == 0.0
