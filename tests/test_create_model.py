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
