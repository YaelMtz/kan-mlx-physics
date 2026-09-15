"""Tests for the Sweep experiment helper."""
import os
import json
import tempfile

import pytest

from kan_mlx_physics import Sweep


def _tmp():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.remove(path)  # Sweep creates it
    return path


def test_cells_cartesian_product():
    sw = Sweep(_tmp(), a=[1, 2], b=["x", "y", "z"])
    cells = sw.cells()
    assert len(cells) == 6
    assert {"a": 1, "b": "x"} in cells
    assert {"a": 2, "b": "z"} in cells


def test_run_records_and_merges_axes():
    path = _tmp()
    sw = Sweep(path, n=[1, 2], seed=range(2))

    @sw.run
    def cell(n, seed):
        return {"val": n * 10 + seed}

    recs = sw.results()
    assert len(recs) == 4
    # axis values are merged into the record automatically
    r = next(r for r in recs if r["n"] == 2 and r["seed"] == 1)
    assert r["val"] == 21
    assert r["fail"] is False
    os.remove(path)


def test_resume_skips_done_cells():
    path = _tmp()
    sw = Sweep(path, seed=range(3))

    @sw.run
    def cell(seed):
        return {"v": seed}

    # second run must not re-execute (would overwrite v with 999)
    @sw.run
    def cell2(seed):
        return {"v": 999}

    assert all(r["v"] != 999 for r in sw.results())
    os.remove(path)


def test_failed_cell_is_recorded_not_fatal():
    path = _tmp()
    sw = Sweep(path, seed=range(3))

    @sw.run
    def cell(seed):
        if seed == 1:
            raise ValueError("boom")
        return {"v": seed}

    recs = {r["seed"]: r for r in sw.results()}
    assert recs[1]["fail"] is True
    assert "boom" in recs[1]["error"]
    assert recs[0]["fail"] is False and recs[2]["v"] == 2
    os.remove(path)


def test_summary_runs(capsys):
    path = _tmp()
    sw = Sweep(path, kind=["a", "b"], seed=range(3))

    @sw.run
    def cell(kind, seed):
        return {"l2": 1.0 if kind == "a" else 5.0}

    sw.summary(group_by=["kind"], show=["l2"])
    out = capsys.readouterr().out
    assert "kind" in out and "l2" in out
    os.remove(path)


def test_requires_at_least_one_axis():
    with pytest.raises(ValueError):
        Sweep(_tmp())


# ---- regression tests for correctness fixes from the library audit ----

def test_predict_with_uncertainty_does_not_corrupt_weights():
    """UQ must restore weights exactly (previously drew mismatched noise)."""
    import mlx.core as mx
    from kan_mlx_physics import MultKAN
    m = MultKAN(width=[2, 5, 1], grid=8, k=3, seed=0)
    if not hasattr(m, "predict_with_uncertainty"):
        import pytest; pytest.skip("no predict_with_uncertainty")
    before = [mx.array(l.coef) for l in m.layers]
    m.predict_with_uncertainty(mx.zeros((4, 2)), n_samples=50, noise_scale=0.1)
    drift = max(float(mx.max(mx.abs(mx.array(l.coef) - b)))
                for l, b in zip(m.layers, before))
    assert drift < 1e-5, f"weights drifted by {drift}"


def test_unknown_potential_raises_not_silent_default():
    """An unregistered function in the equation must error, not silently become
    0.5*x**2 (which would solve a different problem)."""
    import pytest
    from kan_mlx_physics.pde import PDEBuilder
    with pytest.raises((ValueError, Exception)):
        (PDEBuilder("Derivative(u, x, 2) + Vunreg(x)*u = 0")
         .params().domain([0, 1])
         .model(width=[1, 3, 1]).solve(verbose=False))
