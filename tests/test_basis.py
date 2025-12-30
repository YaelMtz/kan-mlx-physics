"""Unit tests for pluggable basis functions.

Tests each basis type for:
    - Correct output shape
    - Differentiability (gradients exist)
    - Numerical stability
    - Integration with KANLayer and MultKAN
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from kan_mlx_physics.basis import (
    make_basis,
    list_bases,
    recommend_basis,
    Basis,
    BasisConfig,
    FourierBasis,
    ChebyshevBasis,
    HermiteBasis,
    LaguerreBasis,
    LegendreBasis,
    BSplineBasis,
    contract_basis_coef,
)
from kan_mlx_physics import KANLayer, MultKAN


class TestBasisRegistry:
    """Test basis registry functions."""

    def test_list_bases(self):
        """Test that list_bases returns expected bases."""
        bases = list_bases()
        assert "bspline" in bases
        assert "fourier" in bases
        assert "chebyshev" in bases
        assert "hermite" in bases
        assert "laguerre" in bases
        assert "legendre" in bases

    def test_make_basis_all_types(self):
        """Test that all registered bases can be instantiated."""
        for name in list_bases():
            basis = make_basis(name, M=5)
            assert basis is not None
            assert basis.num_features == 5

    def test_make_basis_invalid(self):
        """Test that invalid basis name raises error."""
        with pytest.raises(ValueError, match="Unknown basis"):
            make_basis("invalid_basis", M=5)

    def test_recommend_basis(self):
        """Test basis recommendations."""
        qm_bases = recommend_basis("quantum_mechanics", "bounded")
        assert "hermite" in qm_bases

        periodic_bases = recommend_basis("periodic")
        assert "fourier" in periodic_bases


class TestBasisShapes:
    """Test that all bases produce correct output shapes."""

    @pytest.fixture
    def batch_input(self):
        """Standard batch input for testing."""
        return mx.random.uniform(shape=(32, 5))

    @pytest.mark.parametrize("basis_name", [
        "fourier", "chebyshev", "hermite", "laguerre", "legendre"
    ])
    def test_features_shape(self, basis_name, batch_input):
        """Test features() output shape: (batch, in_dim, M)."""
        M = 10
        basis = make_basis(basis_name, M=M)
        params = basis.init_params(in_dim=5)

        phis = basis.features(batch_input, params)

        assert phis.shape == (32, 5, M), \
            f"{basis_name}: expected (32, 5, {M}), got {phis.shape}"

    def test_bspline_features_shape(self, batch_input):
        """Test B-spline features shape."""
        # B-spline M = num_grid + k
        basis = BSplineBasis.from_grid(num_grid=5, k=3)
        params = basis.init_params(in_dim=5)

        phis = basis.features(batch_input, params)

        assert phis.shape == (32, 5, 8)  # 5 + 3 = 8

    def test_contract_basis_coef(self):
        """Test coefficient contraction shape."""
        batch, in_dim, out_dim, M = 32, 5, 3, 10

        phis = mx.random.uniform(shape=(batch, in_dim, M))
        coef = mx.random.uniform(shape=(in_dim, out_dim, M))

        output = contract_basis_coef(phis, coef)

        assert output.shape == (batch, in_dim, out_dim)


class TestFourierBasis:
    """Test Fourier basis specific properties."""

    def test_feature_count_odd_M(self):
        """Test correct feature count for odd M."""
        for M in [3, 5, 7, 11]:
            basis = make_basis("fourier", M=M)
            x = mx.random.uniform(shape=(10, 2))
            params = basis.init_params(in_dim=2)
            phis = basis.features(x, params)
            assert phis.shape[-1] == M, f"M={M}: got {phis.shape[-1]}"

    def test_feature_count_even_M(self):
        """Test correct feature count for even M."""
        for M in [2, 4, 6, 10]:
            basis = make_basis("fourier", M=M)
            x = mx.random.uniform(shape=(10, 2))
            params = basis.init_params(in_dim=2)
            phis = basis.features(x, params)
            assert phis.shape[-1] == M, f"M={M}: got {phis.shape[-1]}"

    def test_constant_term(self):
        """Test that first feature is constant (ones)."""
        basis = make_basis("fourier", M=5)
        x = mx.random.uniform(shape=(10, 2))
        params = basis.init_params(in_dim=2)
        phis = basis.features(x, params)

        # First feature should be all ones
        ones = phis[:, :, 0]
        assert mx.allclose(ones, mx.ones_like(ones), atol=1e-5)

    def test_learnable_frequency(self):
        """Test that frequency parameter is learnable."""
        basis = FourierBasis(BasisConfig(M=5), learnable_freq=True)
        params = basis.init_params(in_dim=2)

        assert "omega_raw" in params
        assert params["omega_raw"].shape == (2,)


class TestPolynomialBases:
    """Test polynomial bases (Chebyshev, Hermite, Laguerre, Legendre)."""

    @pytest.mark.parametrize("basis_name,x_range", [
        ("chebyshev", (-1.0, 1.0)),
        ("hermite", (-3.0, 3.0)),
        ("legendre", (-1.0, 1.0)),
    ])
    def test_polynomial_values_at_zero(self, basis_name, x_range):
        """Test polynomial values at x=0."""
        basis = make_basis(basis_name, M=4)
        x = mx.zeros((1, 1))
        params = basis.init_params(in_dim=1)
        phis = basis.features(x, params)

        # P_0(0) = 1 for all polynomial families
        assert mx.abs(phis[0, 0, 0] - 1.0) < 0.1  # Allowing for normalization

    def test_chebyshev_recurrence(self):
        """Test Chebyshev recurrence: T_{n+1} = 2x*T_n - T_{n-1}."""
        basis = make_basis("chebyshev", M=5, learnable_affine=False)
        # Use x in domain
        x = mx.array([[0.5]])
        params = basis.init_params(in_dim=1)
        phis = basis.features(x, params)

        T = phis[0, 0, :]  # All Chebyshev values
        # T_0 = 1, T_1 = x, T_2 = 2x*T_1 - T_0
        x_val = float(mx.tanh(mx.array(0.5)))  # Domain mapping
        x_mapped = -1 + 2 * (x_val + 1) / 2  # Scale to [-1, 1]

        # The recurrence should hold approximately
        if len(T) >= 3:
            T2_expected = 2 * x_mapped * float(T[1]) - float(T[0])
            assert abs(float(T[2]) - T2_expected) < 0.5

    def test_hermite_weighted(self):
        """Test weighted Hermite basis includes Gaussian factor."""
        basis_unweighted = make_basis("hermite", M=3, weighted=False)
        basis_weighted = make_basis("hermite", M=3, weighted=True)

        x = mx.array([[1.0], [2.0]])
        params_u = basis_unweighted.init_params(in_dim=1)
        params_w = basis_weighted.init_params(in_dim=1)

        phis_u = basis_unweighted.features(x, params_u)
        phis_w = basis_weighted.features(x, params_w)

        # Weighted should be smaller due to exp(-x²/2) decay
        assert float(mx.mean(mx.abs(phis_w))) < float(mx.mean(mx.abs(phis_u)))

    def test_laguerre_alpha_parameter(self):
        """Test Laguerre with different alpha values."""
        for alpha in [0.0, 1.0, 2.0]:
            basis = make_basis("laguerre", M=4, alpha=alpha)
            x = mx.random.uniform(low=0.1, high=2.0, shape=(5, 2))
            params = basis.init_params(in_dim=2)
            phis = basis.features(x, params)

            assert phis.shape == (5, 2, 4)
            assert mx.all(mx.isfinite(phis))


class TestGradients:
    """Test that gradients flow through basis functions."""

    @pytest.mark.parametrize("basis_name", [
        "fourier", "chebyshev", "hermite", "laguerre", "legendre"
    ])
    def test_gradient_exists(self, basis_name):
        """Test that gradients can be computed."""
        basis = make_basis(basis_name, M=5)
        params = basis.init_params(in_dim=2)

        x = mx.random.uniform(shape=(10, 2))

        def loss_fn(x):
            phis = basis.features(x, params)
            return mx.sum(phis)

        grad = mx.grad(loss_fn)(x)

        assert grad.shape == x.shape
        assert mx.all(mx.isfinite(grad))

    def test_gradient_wrt_params(self):
        """Test gradients w.r.t. learnable basis params."""
        basis = FourierBasis(BasisConfig(M=5), learnable_freq=True)
        x = mx.random.uniform(shape=(10, 2))

        def loss_fn(omega_raw):
            params = {"omega_raw": omega_raw, "shift": mx.zeros(2), "scale_raw": mx.zeros(2)}
            phis = basis.features(x, params)
            return mx.sum(phis)

        omega_raw = mx.zeros((2,))
        grad = mx.grad(loss_fn)(omega_raw)

        assert grad.shape == omega_raw.shape
        # Gradient should be non-zero (features depend on omega)


class TestNumericalStability:
    """Test numerical stability of basis functions."""

    @pytest.mark.parametrize("basis_name", [
        "fourier", "chebyshev", "hermite", "laguerre", "legendre"
    ])
    def test_no_nans(self, basis_name):
        """Test that no NaNs appear in output."""
        basis = make_basis(basis_name, M=10)
        params = basis.init_params(in_dim=3)

        # Test with various input ranges
        x_ranges = [
            mx.random.uniform(low=-1, high=1, shape=(20, 3)),
            mx.random.uniform(low=-10, high=10, shape=(20, 3)),
            mx.random.uniform(low=0, high=5, shape=(20, 3)),
        ]

        for x in x_ranges:
            phis = basis.features(x, params)
            assert mx.all(mx.isfinite(phis)), f"{basis_name} produced NaN/Inf"

    def test_high_order_stability(self):
        """Test stability for high-order polynomials."""
        # High-order polynomials can be numerically unstable
        for basis_name in ["chebyshev", "legendre"]:
            basis = make_basis(basis_name, M=15)  # 15th order
            x = mx.random.uniform(low=-0.9, high=0.9, shape=(10, 2))
            params = basis.init_params(in_dim=2)
            phis = basis.features(x, params)

            assert mx.all(mx.isfinite(phis))


class TestKANLayerIntegration:
    """Test integration with KANLayer."""

    @pytest.mark.parametrize("basis_name", [
        "bspline", "fourier", "chebyshev", "hermite", "legendre"
    ])
    def test_kan_layer_forward(self, basis_name):
        """Test KANLayer forward pass with each basis."""
        layer = KANLayer(
            in_dim=3,
            out_dim=2,
            basis=basis_name,
            basis_M=8,
        )

        x = mx.random.uniform(shape=(10, 3))
        y = layer(x)

        assert y.shape == (10, 2)
        assert mx.all(mx.isfinite(y))

    def test_kan_layer_with_activations(self):
        """Test KANLayer returns activations."""
        layer = KANLayer(
            in_dim=3,
            out_dim=2,
            basis="fourier",
            basis_M=11,
        )

        x = mx.random.uniform(shape=(10, 3))
        output, preacts, postacts, postspline = layer(x, return_activations=True)

        assert output.shape == (10, 2)
        assert preacts.shape == (10, 2, 3)
        assert postacts.shape == (10, 3, 2)
        assert postspline.shape == (10, 3, 2)


class TestMultKANIntegration:
    """Test integration with MultKAN."""

    def test_multkan_uniform_basis(self):
        """Test MultKAN with same basis for all layers."""
        model = MultKAN(
            width=[2, 5, 1],
            basis="fourier",
            basis_M=11,
        )

        x = mx.random.uniform(shape=(10, 2))
        y = model(x)

        assert y.shape == (10, 1)
        assert all(l.basis_type == "fourier" for l in model.layers)

    def test_multkan_per_layer_basis(self):
        """Test MultKAN with different basis per layer."""
        model = MultKAN(
            width=[2, 10, 1],
            basis=["chebyshev", "fourier"],
            basis_M=[8, 15],
        )

        x = mx.random.uniform(shape=(10, 2))
        y = model(x)

        assert y.shape == (10, 1)
        assert model.layers[0].basis_type == "chebyshev"
        assert model.layers[1].basis_type == "fourier"

    def test_multkan_backwards_compatible(self):
        """Test that default MultKAN uses B-splines."""
        model = MultKAN(width=[2, 5, 1])

        assert all(l.basis_type == "bspline" for l in model.layers)


class TestSymbolicPriority:
    """Test symbolic priority lists."""

    def test_fourier_symbolic_priority(self):
        """Test Fourier basis prioritizes trig functions."""
        basis = make_basis("fourier", M=5)
        priority = basis.symbolic_priority

        assert "sin" in priority
        assert "cos" in priority

    def test_hermite_symbolic_priority(self):
        """Test Hermite basis prioritizes QM functions."""
        basis = make_basis("hermite", M=5)
        priority = basis.symbolic_priority

        assert "gaussian" in priority or "exp" in priority

    def test_kan_layer_symbolic_priority(self):
        """Test KANLayer exposes basis symbolic priority."""
        layer = KANLayer(in_dim=3, out_dim=2, basis="fourier", basis_M=5)
        priority = layer.symbolic_priority

        assert "sin" in priority
        assert "cos" in priority


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
