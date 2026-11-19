"""Tests for _create_model parameter forwarding (base_fun, noise_scale, seed)."""

from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest


def _make_analysis():
    """Minimal PDEAnalysis mock — only the fields _create_model reads."""
    analysis = MagicMock()
    analysis.recommended_basis = "bspline"
    analysis.is_eigenvalue_problem = False
    return analysis


def _make_domain():
    """1D domain via PDEBuilder's Domain dataclass."""
    from kan_mlx_physics.pde.dsl import Domain
    return Domain(bounds=[(0.0, 1.0)])


# Always supply basis= so analysis.recommended_basis is never touched.
_BASE_CFG = {"width": [1, 1], "basis": "bspline"}


class TestBaseFunForwarding:
    """base_fun is forwarded from config to MultKAN."""

    def test_callable_forwarded(self):
        from kan_mlx_physics.pde.dsl import _create_model

        cfg = {**_BASE_CFG, "base_fun": lambda x: x}
        model = _create_model(cfg, _make_domain(), _make_analysis())

        x = mx.array([[2.0]])
        result = model.layers[0].base_fun(x)
        mx.eval(result)
        assert float(result[0, 0]) == pytest.approx(2.0)

    def test_string_identity_forwarded(self):
        from kan_mlx_physics.pde.dsl import _create_model

        cfg = {**_BASE_CFG, "base_fun": "identity"}
        model = _create_model(cfg, _make_domain(), _make_analysis())

        x = mx.array([[3.0]])
        result = model.layers[0].base_fun(x)
        mx.eval(result)
        assert float(result[0, 0]) == pytest.approx(3.0)

    def test_string_silu_forwarded(self):
        from kan_mlx_physics.pde.dsl import _create_model

        cfg = {**_BASE_CFG, "base_fun": "silu"}
        model = _create_model(cfg, _make_domain(), _make_analysis())

        x = mx.array([[1.0]])
        result = model.layers[0].base_fun(x)
        expected = nn.silu(x)
        mx.eval(result, expected)
        assert float(result[0, 0]) == pytest.approx(float(expected[0, 0]), rel=1e-5)

    def test_string_tanh_forwarded(self):
        from kan_mlx_physics.pde.dsl import _create_model

        cfg = {**_BASE_CFG, "base_fun": "tanh"}
        model = _create_model(cfg, _make_domain(), _make_analysis())

        x = mx.array([[1.0]])
        result = model.layers[0].base_fun(x)
        expected = mx.tanh(x)
        mx.eval(result, expected)
        assert float(result[0, 0]) == pytest.approx(float(expected[0, 0]), rel=1e-5)

    def test_unknown_string_raises(self):
        from kan_mlx_physics.pde.dsl import _create_model

        cfg = {**_BASE_CFG, "base_fun": "sigmoid"}
        with pytest.raises(ValueError, match="Unknown base_fun string"):
            _create_model(cfg, _make_domain(), _make_analysis())

    def test_default_is_silu(self):
        """When base_fun is absent from config, default is nn.silu."""
        from kan_mlx_physics.pde.dsl import _create_model

        model = _create_model(dict(_BASE_CFG), _make_domain(), _make_analysis())

        x = mx.array([[1.0]])
        result = model.layers[0].base_fun(x)
        expected = nn.silu(x)
        mx.eval(result, expected)
        assert float(result[0, 0]) == pytest.approx(float(expected[0, 0]), rel=1e-5)


class TestSeedForwarding:
    """seed is forwarded so initialization is reproducible."""

    def test_same_seed_same_weights(self):
        from kan_mlx_physics.pde.dsl import _create_model

        cfg = {"width": [1, 5, 1], "basis": "bspline", "seed": 42}
        m1 = _create_model(dict(cfg), _make_domain(), _make_analysis())
        m2 = _create_model(dict(cfg), _make_domain(), _make_analysis())

        w1 = m1.layers[0].coef
        w2 = m2.layers[0].coef
        mx.eval(w1, w2)
        assert mx.allclose(w1, w2).item()

    def test_different_seeds_different_weights(self):
        from kan_mlx_physics.pde.dsl import _create_model

        m1 = _create_model(
            {"width": [1, 5, 1], "basis": "bspline", "seed": 1},
            _make_domain(), _make_analysis(),
        )
        m2 = _create_model(
            {"width": [1, 5, 1], "basis": "bspline", "seed": 2},
            _make_domain(), _make_analysis(),
        )

        w1 = m1.layers[0].coef
        w2 = m2.layers[0].coef
        mx.eval(w1, w2)
        assert not mx.allclose(w1, w2).item()


class TestNoiseScaleForwarding:
    """noise_scale is forwarded to MultKAN."""

    def test_noise_scale_stored(self):
        from kan_mlx_physics.pde.dsl import _create_model

        cfg = {**_BASE_CFG, "noise_scale": 0.42}
        model = _create_model(cfg, _make_domain(), _make_analysis())
        assert model.noise_scale == pytest.approx(0.42)

    def test_noise_scale_default(self):
        from kan_mlx_physics.pde.dsl import _create_model

        model = _create_model(dict(_BASE_CFG), _make_domain(), _make_analysis())
        assert model.noise_scale == pytest.approx(0.1)


def test_basis_none_defaults_to_bspline():
    """Regression: MultKAN(basis=None) must not crash — None means default B-spline.
    (Previously `len(basis)` raised TypeError on None.)"""
    import mlx.core as mx
    from kan_mlx_physics import MultKAN
    m = MultKAN(width=[2, 5, 1], basis=None, grid=8, k=3)
    y = m(mx.zeros((4, 2)))
    assert y.shape == (4, 1)


def _trainable_leaf_keys(tree, prefix=""):
    """Flatten an MLX parameter tree to dotted leaf keys (for freeze assertions)."""
    keys = []
    if isinstance(tree, dict):
        for k, v in tree.items():
            keys += _trainable_leaf_keys(v, f"{prefix}{k}.")
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            keys += _trainable_leaf_keys(v, f"{prefix}{i}.")
    else:
        keys.append(prefix.rstrip("."))
    return keys


class TestEdgeCoefPinningAndFreezing:
    """init_edge_coef sets edge coefficients; freeze_edges removes them from training.

    Supports solves where an edge shape is fixed by construction (the architectural
    ★-ladder: a one-hot read-out of a physically-associated basis) and only the edge
    affine and trainable scalars are optimized.
    """

    def _lag_cfg(self, M=6, init=None, freeze=False):
        cfg = {
            "width": [1, 1], "basis": "laguerre", "basis_M": M, "grid": 10, "k": 3,
            "basis_kwargs": {"alpha": 0.0, "weighted": True,
                             "nonnegative_input": True, "fixed_scale": 2.0},
            "base_fun": (lambda x: mx.zeros_like(x)), "noise_scale": 0.0,
            "grid_range": (0.0, 10.0), "seed": 5,
        }
        if init is not None:
            cfg["init_edge_coef"] = init
        if freeze:
            cfg["freeze_edges"] = True
        return cfg

    def test_init_edge_coef_array_sets_layer0(self):
        from kan_mlx_physics.pde.dsl import _create_model
        M = 6
        oh = mx.zeros((1, 1, M))
        oh[0, 0, 3] = 1.0
        model = _create_model(self._lag_cfg(M=M, init=oh), _make_domain(), _make_analysis())
        got = model.layers[0].coef
        mx.eval(got)
        assert got.shape == (1, 1, M)
        assert float(got[0, 0, 3]) == pytest.approx(1.0)
        assert float(got[0, 0, 0]) == pytest.approx(0.0)

    def test_init_edge_coef_dict_selects_layer(self):
        from kan_mlx_physics.pde.dsl import _create_model
        M = 6
        oh = mx.zeros((1, 1, M)); oh[0, 0, 2] = 0.7
        cfg = self._lag_cfg(M=M, init={0: oh})
        model = _create_model(cfg, _make_domain(), _make_analysis())
        assert float(model.layers[0].coef[0, 0, 2]) == pytest.approx(0.7)

    def test_init_edge_coef_callable(self):
        from kan_mlx_physics.pde.dsl import _create_model
        M = 6

        def f(i, layer):
            v = mx.zeros(layer.coef.shape)
            v[0, 0, 1] = -0.5
            return v

        model = _create_model(self._lag_cfg(M=M, init=f), _make_domain(), _make_analysis())
        assert float(model.layers[0].coef[0, 0, 1]) == pytest.approx(-0.5)

    def test_freeze_edges_removes_coef_from_trainables(self):
        from kan_mlx_physics.pde.dsl import _create_model
        model = _create_model(self._lag_cfg(freeze=True), _make_domain(), _make_analysis())
        keys = _trainable_leaf_keys(model.trainable_parameters())
        assert not any(k.endswith("coef") for k in keys), f"coef still trainable: {keys}"
        # the edge affine must remain trainable
        assert any("basis_scale_raw" in k or "basis_shift" in k for k in keys)

    def test_unfrozen_keeps_coef_trainable(self):
        from kan_mlx_physics.pde.dsl import _create_model
        model = _create_model(self._lag_cfg(freeze=False), _make_domain(), _make_analysis())
        keys = _trainable_leaf_keys(model.trainable_parameters())
        assert any(k.endswith("coef") for k in keys)

    def test_frozen_coef_unchanged_by_gradient_update(self):
        """A frozen one-hot coef must survive an optimizer step unchanged."""
        import mlx.optimizers as optim
        from kan_mlx_physics.pde.dsl import _create_model
        M = 6
        oh = mx.zeros((1, 1, M)); oh[0, 0, 3] = 1.0
        model = _create_model(self._lag_cfg(M=M, init=oh, freeze=True),
                              _make_domain(), _make_analysis())
        before = mx.array(model.layers[0].coef)
        opt = optim.Adam(learning_rate=1e-1)

        def loss_fn(m):
            s = mx.linspace(0.1, 9.0, 64).reshape(-1, 1)
            return mx.mean(m(s) ** 2)

        lg = nn.value_and_grad(model, loss_fn)
        _, grads = lg(model)
        opt.update(model, grads)
        mx.eval(model.parameters())
        after = model.layers[0].coef
        assert float(mx.max(mx.abs(after - before))) == pytest.approx(0.0, abs=1e-7)


class TestEdgeAffinePinningAndMultiKeyFreeze:
    """init_edge_affine pins the edge affine; freeze_edges accepts a key list.

    Together with init_edge_coef this pins the full edge SHAPE (coefficients AND affine),
    which the architectural ★-ladder needs: a weighted-Laguerre edge with identity affine
    and fixed_scale=2 emits exactly L_m(2s)e^{-s}, so a one-hot at index n is the n-th
    Wigner ★-genfunction and must stay fixed while only E trains.
    """

    def _cfg(self, M=8, n=3):
        import numpy as np
        oh = np.zeros((1, 1, M), dtype=np.float32)
        oh[0, 0, n] = (-1.0) ** n / np.pi
        return {
            "width": [1, 1], "basis": "laguerre", "basis_M": M, "grid": 28, "k": 3,
            "basis_kwargs": {"alpha": 0.0, "weighted": True,
                             "nonnegative_input": True, "fixed_scale": 2.0},
            "base_fun": (lambda x: mx.zeros_like(x)), "noise_scale": 0.0,
            "grid_range": (0.0, 10.0), "seed": 5,
            "init_edge_coef": oh, "init_edge_affine": "identity",
            # freeze the shape (coef + affine + scale_base); leave scale_sp as the amplitude
            "freeze_edges": ["coef", "basis_scale_raw", "basis_shift", "scale_base"],
        }

    def test_identity_affine_sets_scale_one_shift_zero(self):
        import math
        from kan_mlx_physics.pde.dsl import _create_model
        model = _create_model(self._cfg(), _make_domain(), _make_analysis())
        layer = model.layers[0]
        mx.eval(layer.basis_shift, layer.basis_scale_raw)
        scale = math.log(1 + math.exp(float(layer.basis_scale_raw.reshape(-1)[0])))
        assert float(layer.basis_shift.reshape(-1)[0]) == pytest.approx(0.0, abs=1e-5)
        assert scale == pytest.approx(1.0, abs=1e-4)

    def test_edge_emits_exact_wigner_genfunction(self):
        """With one-hot coef + identity affine, model(s) == ((-1)^n/pi) L_n(2s) e^{-s}."""
        import numpy as np
        from scipy.special import eval_laguerre
        from kan_mlx_physics.pde.dsl import _create_model
        n = 3
        model = _create_model(self._cfg(n=n), _make_domain(), _make_analysis())
        s = np.linspace(0, 10, 500).astype(np.float32)
        out = np.array(model(mx.array(s.reshape(-1, 1)))).ravel()
        Wn = (-1) ** n / np.pi * eval_laguerre(n, 2 * s) * np.exp(-s)
        assert np.max(np.abs(out - Wn)) < 1e-5

    def test_multikey_freeze_leaves_only_amplitude(self):
        from kan_mlx_physics.pde.dsl import _create_model
        model = _create_model(self._cfg(), _make_domain(), _make_analysis())
        keys = _trainable_leaf_keys(model.trainable_parameters())
        # shape leaves are frozen
        for frozen in ("coef", "basis_scale_raw", "basis_shift", "scale_base"):
            assert not any(k.endswith(frozen) for k in keys), f"{frozen} still trainable"
        # the amplitude (scale_sp) remains trainable — the one physical DOF trace pins
        assert any(k.endswith("scale_sp") for k in keys)

    def test_shape_survives_training_up_to_amplitude(self):
        """With the shape frozen, training moves only the amplitude: out stays ∝ W_n."""
        import numpy as np
        import mlx.optimizers as optim
        from scipy.special import eval_laguerre
        from kan_mlx_physics.pde.dsl import _create_model
        n = 3
        model = _create_model(self._cfg(n=n), _make_domain(), _make_analysis())
        opt = optim.Adam(learning_rate=5e-2)

        def loss_fn(m):
            s = mx.linspace(0.1, 9.0, 64).reshape(-1, 1)
            return mx.mean((m(s) - 0.5) ** 2)   # arbitrary pull away from W_n

        lg = nn.value_and_grad(model, loss_fn)
        for _ in range(10):
            _, grads = lg(model)
            opt.update(model, grads)
            mx.eval(model.parameters())
        s = np.linspace(0, 10, 500).astype(np.float32)
        out = np.array(model(mx.array(s.reshape(-1, 1)))).ravel()
        Wn = (-1) ** n / np.pi * eval_laguerre(n, 2 * s) * np.exp(-s)
        A = float(np.array(model.layers[0].scale_sp).reshape(-1)[0])
        # shape unmoved up to the learnable amplitude A = scale_sp
        assert np.max(np.abs(out - A * Wn)) < 1e-4
