"""Round-trip tests for MultKAN.saveckpt / loadckpt.

Regression guard for the checkpointing fix: before it, a non-B-spline model
(Laguerre/Hermite/mixed) was saved with only width/grid/k config and B-spline
coefficients, so it reloaded as a *default B-spline* — silently losing the basis
configuration and every basis parameter. Trainable params (the eigenvalue E)
were not persisted at all.

These tests assert exact reproduction of the forward pass after a round-trip
through a *fresh, differently-shaped* model (the realistic reload scenario).
"""

import os
import pickle
import tempfile

import mlx.core as mx
import numpy as np
import pytest

from kan_mlx_physics import MultKAN, register_physics_symbolic

register_physics_symbolic()


def _zeros(x):
    return mx.zeros_like(x)


def _max_abs_diff(model_a, model_b, x):
    ya = np.array(model_a(x)).flatten()
    yb = np.array(model_b(x)).flatten()
    return float(np.max(np.abs(ya - yb)))


class TestCheckpointRoundTrip:
    def test_laguerre_roundtrip_reproduces_forward(self, tmp_path):
        """A weighted-Laguerre model must reload identically, not as B-spline."""
        m = MultKAN(
            width=[1, 1], grid=20, k=3, basis="laguerre",
            basis_kwargs={"alpha": 0.0, "weighted": True,
                          "nonnegative_input": True, "fixed_scale": 2.0},
            base_fun=_zeros, noise_scale=0.1, grid_range=(0, 6.0), seed=42,
        )
        x = mx.linspace(0, 6, 50).reshape(-1, 1)
        p = str(tmp_path / "lag")
        m.saveckpt(p, extra={"trainable_params": {"E": 0.5001}})

        # Fresh model with a DIFFERENT default architecture (B-spline).
        m2 = MultKAN(width=[2, 2], grid=5, k=3)
        extra = m2.loadckpt(p)

        assert m2.layers[0].basis_type == "laguerre"
        assert _max_abs_diff(m, m2, x) < 1e-5
        assert extra == {"trainable_params": {"E": 0.5001}}

    def test_bspline_roundtrip_no_regression(self, tmp_path):
        """The classic B-spline path must still round-trip exactly."""
        m = MultKAN(width=[1, 3, 1], grid=5, k=3, seed=7)
        x = mx.linspace(-1, 1, 40).reshape(-1, 1)
        p = str(tmp_path / "bs")
        m.saveckpt(p)

        m2 = MultKAN(width=[2, 2], grid=5, k=3)
        assert m2.loadckpt(p) is None  # no extra saved
        assert _max_abs_diff(m, m2, x) < 1e-5

    def test_hermite_roundtrip(self, tmp_path):
        """Another polynomial basis with its own params (basis_shift etc.)."""
        m = MultKAN(
            width=[1, 1], grid=8, k=3, basis="hermite",
            base_fun=_zeros, grid_range=(-3.0, 3.0), seed=1,
        )
        x = mx.linspace(-3, 3, 40).reshape(-1, 1)
        p = str(tmp_path / "herm")
        m.saveckpt(p)
        m2 = MultKAN(width=[2, 2], grid=5, k=3)
        m2.loadckpt(p)
        assert m2.layers[0].basis_type == "hermite"
        assert _max_abs_diff(m, m2, x) < 1e-5

    def test_extra_trainable_params_restored(self, tmp_path):
        """The `extra` payload (e.g. eigenvalue E) round-trips."""
        m = MultKAN(width=[1, 1], grid=5, k=3, seed=0)
        p = str(tmp_path / "extra")
        m.saveckpt(p, extra={"trainable_params": {"E": 1.5}, "note": "n=1"})
        m2 = MultKAN(width=[1, 1], grid=5, k=3)
        extra = m2.loadckpt(p)
        assert extra["trainable_params"]["E"] == 1.5
        assert extra["note"] == "n=1"

    def test_legacy_format_still_loads(self, tmp_path):
        """Old checkpoints (width/grid/k + coef only, no version key) load."""
        m = MultKAN(width=[1, 3, 1], grid=5, k=3, seed=7)
        x = mx.linspace(-1, 1, 40).reshape(-1, 1)

        # Hand-author a legacy checkpoint in the pre-fix format.
        p = str(tmp_path / "legacy")
        old = {}
        for i, layer in enumerate(m.layers):
            old[f"layer_{i}_coef"] = np.array(layer.coef)
            old[f"layer_{i}_scale_sp"] = np.array(layer.scale_sp)
            old[f"layer_{i}_scale_base"] = np.array(layer.scale_base)
            old[f"layer_{i}_grid"] = np.array(layer.grid)
            old[f"layer_{i}_mask"] = np.array(layer.mask)
        for i, sym in enumerate(m.symbolic_funs):
            old[f"sym_{i}_fns_name"] = sym.fns_name
            for c in "abcd":
                old[f"sym_{i}_affine_{c}"] = np.array(getattr(sym, f"affine_{c}"))
        with open(p + "_params.pkl", "wb") as f:
            pickle.dump(old, f)
        with open(p + "_config.pkl", "wb") as f:
            pickle.dump({"width": [1, 3, 1], "grid": 5, "k": 3}, f)

        m2 = MultKAN(width=[2, 2], grid=5, k=3)
        assert m2.loadckpt(p) is None
        assert _max_abs_diff(m, m2, x) < 1e-5
