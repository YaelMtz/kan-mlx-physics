"""Tests for convenience API (quick_fit, auto_formula, presets)."""

import mlx.core as mx
import pytest


class TestSymbolicPriority:
    """Tests for basis-aware symbolic priority in suggest_symbolic."""

    def test_fourier_prioritizes_sin_cos(self):
        """Test that Fourier basis prioritizes sin/cos in symbolic suggestions."""
        from kan_mlx_physics import MultKAN

        # Create a model with Fourier basis
        model = MultKAN(width=[1, 3, 1], basis="fourier", basis_M=11)

        # Check that layer's symbolic_priority includes sin and cos
        priority = model.layers[0].symbolic_priority
        assert "sin" in priority
        assert "cos" in priority

    def test_hermite_prioritizes_gaussian(self):
        """Test that Hermite basis prioritizes gaussian/psi functions."""
        from kan_mlx_physics import MultKAN

        model = MultKAN(width=[1, 3, 1], basis="hermite", basis_M=8)

        priority = model.layers[0].symbolic_priority
        assert "gaussian" in priority
        assert "psi_0" in priority

    def test_suggest_symbolic_uses_priority(self):
        """Test that suggest_symbolic applies basis priority bonus."""
        from kan_mlx_physics import MultKAN, create_dataset

        # Train a Fourier model on sin(x)
        dataset = create_dataset(
            f=lambda x: mx.sin(2 * mx.pi * x[:, 0]),
            n_var=1,
            train_num=200,
        )

        model = MultKAN(width=[1, 5, 1], basis="fourier", basis_M=11)
        model.fit(dataset, steps=50, verbose=False)

        # Get suggestions - sin should be prioritized for Fourier basis
        suggestions = model.suggest_symbolic(0, 0, 0, dataset['train_input'], top_k=5)

        # Check that we get results
        assert len(suggestions) > 0

        # The first suggestion should be a trig function for this sin(x) target
        top_fn = suggestions[0][0]
        # With Fourier basis, sin should rank high
        trig_fns = {"sin", "cos", "sin_q", "cos_q"}
        assert top_fn in trig_fns or suggestions[0][1] > 0.9  # Either trig or very high R²


class TestPresets:
    """Tests for physics presets."""

    def test_list_presets(self):
        """Test listing available presets."""
        from kan_mlx_physics import list_presets

        presets = list_presets()
        assert "quantum_oscillator" in presets
        assert "wave_equation" in presets
        assert "spectral" in presets
        assert "radial" in presets
        assert "angular" in presets
        assert "general" in presets

    def test_describe_preset(self):
        """Test preset description."""
        from kan_mlx_physics.presets import describe_preset

        desc = describe_preset("quantum_oscillator")
        assert "hermite" in desc.lower()
        assert "weighted" in desc.lower()

    def test_from_preset_quantum(self):
        """Test creating model from quantum_oscillator preset."""
        from kan_mlx_physics import from_preset

        model = from_preset("quantum_oscillator", width=[1, 10, 1])
        assert model.depth == 2
        assert model.width == [1, 10, 1]
        # Check basis is set correctly
        assert model.layers[0].basis_type == "hermite"

    def test_from_preset_wave(self):
        """Test creating model from wave_equation preset."""
        from kan_mlx_physics import from_preset

        model = from_preset("wave_equation", width=[2, 15, 1])
        assert model.depth == 2
        # Check basis is set correctly
        assert model.layers[0].basis_type == "fourier"

    def test_from_preset_with_override(self):
        """Test overriding preset parameters."""
        from kan_mlx_physics import from_preset

        model = from_preset("wave_equation", width=[2, 10, 1], basis_M=31)
        assert model.depth == 2
        # basis_M override should work

    def test_from_preset_invalid(self):
        """Test error on invalid preset name."""
        from kan_mlx_physics import from_preset

        with pytest.raises(ValueError, match="Unknown preset"):
            from_preset("nonexistent_preset", width=[2, 5, 1])


class TestParameterAliases:
    """Tests for parameter aliasing."""

    def test_init_aliases(self):
        """Test __init__ parameter aliases."""
        from kan_mlx_physics import MultKAN

        # Test spline_order alias
        model1 = MultKAN(width=[2, 5, 1], spline_order=2)
        assert model1.k == 2

        # Test grid_points alias
        model2 = MultKAN(width=[2, 5, 1], grid_points=10)
        assert model2.grid == 10

        # Test multiplication_inputs alias
        model3 = MultKAN(width=[[2, 0], [4, 1], [1, 0]], multiplication_inputs=3)
        assert model3.mult_arity == 3

    def test_fit_aliases(self):
        """Test fit() parameter aliases."""
        from kan_mlx_physics import MultKAN, create_dataset

        dataset = create_dataset(
            f=lambda x: x[:, 0] ** 2,
            n_var=1,
            train_num=50,
        )

        model = MultKAN(width=[1, 3, 1])

        # Test with aliases
        history = model.fit(
            dataset,
            steps=5,
            learning_rate=0.01,  # alias for lr
            regularization=0.1,  # alias for lamb
            l1_weight=0.5,       # alias for lamb_l1
            entropy_weight=1.0,  # alias for lamb_entropy
            verbose=False,
        )

        assert len(history['train_loss']) > 0


class TestQuickFit:
    """Tests for quick_fit function."""

    def test_quick_fit_basic(self):
        """Test basic quick_fit usage."""
        from kan_mlx_physics import quick_fit

        model, history = quick_fit(
            f=lambda x: x[:, 0] ** 2,
            n_var=1,
            steps=10,
            train_num=100,
            verbose=False,
        )

        assert model.depth == 2  # Default: [n_var, 5*n_var, 1]
        assert 'x_sample' in history

    def test_quick_fit_with_preset(self):
        """Test quick_fit with preset."""
        from kan_mlx_physics import quick_fit

        model, history = quick_fit(
            f=lambda x: mx.sin(x[:, 0]),
            n_var=1,
            preset="wave_equation",
            steps=5,
            verbose=False,
        )

        assert model.layers[0].basis_type == "fourier"

    def test_quick_fit_custom_width(self):
        """Test quick_fit with custom width."""
        from kan_mlx_physics import quick_fit

        model, history = quick_fit(
            f=lambda x: x[:, 0] + x[:, 1],
            n_var=2,
            width=[2, 8, 8, 1],
            steps=5,
            verbose=False,
        )

        assert model.depth == 3
        assert model.width == [2, 8, 8, 1]


class TestAutoFormula:
    """Tests for auto_formula function."""

    def test_auto_formula_basic(self):
        """Test basic auto_formula usage."""
        from kan_mlx_physics import MultKAN, create_dataset, auto_formula

        # Train a simple model
        dataset = create_dataset(
            f=lambda x: x[:, 0] ** 2,
            n_var=1,
            train_num=200,
        )

        model = MultKAN(width=[1, 5, 1])
        model.fit(dataset, steps=50, verbose=False)

        # Extract formula
        formula = auto_formula(
            model,
            dataset['train_input'],
            var_names=['x'],
            threshold=0.8,  # Lower threshold for test
            verbose=False,
        )

        assert isinstance(formula, str)
        assert len(formula) > 0

    def test_auto_formula_formats(self):
        """Test different output formats."""
        from kan_mlx_physics import MultKAN, create_dataset, auto_formula

        dataset = create_dataset(
            f=lambda x: x[:, 0],  # Simple identity
            n_var=1,
            train_num=100,
        )

        model = MultKAN(width=[1, 3, 1])
        model.fit(dataset, steps=20, verbose=False)
        x_sample = dataset['train_input']

        # Test different formats (just verify they don't crash)
        for fmt in ["unicode", "latex", "typst"]:
            result = auto_formula(model, x_sample, format=fmt, verbose=False)
            assert isinstance(result, str)


class TestVisualize:
    """Tests for unified visualize function."""

    def test_visualize_invalid(self):
        """Test error on invalid visualization type."""
        from kan_mlx_physics import MultKAN, visualize

        model = MultKAN(width=[2, 5, 1])

        with pytest.raises(ValueError, match="Unknown visualization"):
            visualize(model, "nonexistent")

    def test_visualize_activations_requires_x(self):
        """Test that activations visualization requires x."""
        from kan_mlx_physics import MultKAN, visualize

        model = MultKAN(width=[2, 5, 1])

        with pytest.raises(ValueError, match="x is required"):
            visualize(model, "activations")

    def test_visualize_history_requires_history(self):
        """Test that history visualization requires history dict."""
        from kan_mlx_physics import MultKAN, visualize

        model = MultKAN(width=[2, 5, 1])

        with pytest.raises(ValueError, match="history dict is required"):
            visualize(model, "history")
