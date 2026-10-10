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


def _tiny_kan():
    import mlx.core as mx
    from kan_mlx_physics import MultKAN, register_physics_symbolic
    register_physics_symbolic()
    m = MultKAN(width=[1, 1], grid=8, k=3, basis="laguerre",
                basis_kwargs={"weighted": True, "nonnegative_input": True,
                              "fixed_scale": 2.0},
                base_fun=lambda z: mx.zeros_like(z), seed=0)
    x = mx.linspace(0, 6, 80).reshape(-1, 1)
    m(x)
    return m, x


def test_paper_network_saves():
    from kan_mlx_physics.viz import paper
    paper.use()
    m, _ = _tiny_kan()
    p = _tmp()
    paper.network(m, in_vars=["r2"], out_vars=["W"]).save(p)
    assert os.path.getsize(p) > 0; os.remove(p)


def test_paper_symbolic_refinement_saves_and_has_formula():
    from kan_mlx_physics.viz import paper
    paper.use()
    m, x = _tiny_kan()
    p = _tmp()
    ref = paper.symbolic_refinement(m, x, allowed_fns={"exp", "L_0"},
                                    var_names=["r2"], in_vars=["r2"],
                                    out_vars=["W"])
    ref.save(p)
    assert os.path.getsize(p) > 0
    assert isinstance(ref.formula, str) and len(ref.formula) > 0
    os.remove(p)
