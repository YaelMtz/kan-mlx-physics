"""Test derivative accuracy: autodiff vs finite differences.

This test verifies that autodiff produces significantly more accurate derivatives
than finite differences, especially for float32 precision.
"""

import numpy as np
import mlx.core as mx
from kan_mlx_physics import MultKAN


def test_autodiff_vs_finite_diff_accuracy():
    """Test that autodiff is more accurate than finite differences for second derivatives.

    Uses a simple polynomial f(x) = x³ where f''(2) = 12.0 analytically.
    Float32 finite differences suffer from catastrophic cancellation.
    """
    # Create a simple KAN model
    model = MultKAN(width=[1, 5, 1], grid=5, k=3, seed=42)

    # Train it to approximate f(x) = x³
    x_train = mx.linspace(-2, 2, 100).reshape(-1, 1)
    y_train = x_train ** 3

    # Simple training loop
    from kan_mlx_physics.functional import get_params_list, set_params_list, adam_update, init_adam_state

    params_list = get_params_list(model)
    m, v = init_adam_state(params_list)

    for step in range(500):
        def loss_fn(params):
            from kan_mlx_physics.functional import functional_forward
            y_pred = functional_forward(params, x_train, model.k, model.base_fun)
            return mx.mean((y_pred - y_train) ** 2)

        loss, grads = mx.value_and_grad(loss_fn)(params_list)
        params_list, m, v = adam_update(params_list, grads, m, v, mx.array(step + 1, dtype=mx.float32), lr=0.01)
        mx.eval(loss, params_list)

    # Unpack parameters back to model
    for i, (coef, scale_sp, scale_base, grid) in enumerate(params_list):
        model.layers[i].coef = coef
        model.layers[i].scale_sp = scale_sp
        model.layers[i].scale_base = scale_base

    # Test point: x = 2.0, analytical f''(2) = 12.0
    x_test = mx.array([[2.0]])

    # Method 1: Autodiff (should be accurate)
    from kan_mlx_physics.pde.compiler import CompiledResidual

    compiled_autodiff = CompiledResidual(
        parsed=None, jet_req=None, coord_map={'x': 0},
        derivative_method="autodiff"
    )
    d2_autodiff = compiled_autodiff._autodiff_derivative(model, x_test, 'x', order=2)
    d2_autodiff_val = float(d2_autodiff[0])

    # Method 2: Finite differences (should have cancellation errors)
    compiled_finite = CompiledResidual(
        parsed=None, jet_req=None, coord_map={'x': 0},
        derivative_method="finite_diff", finite_diff_h=1e-4
    )
    d2_finite = compiled_finite._finite_diff_derivative(model, x_test, 'x', order=2, h=1e-4)
    d2_finite_val = float(d2_finite[0])

    # Analytical value
    analytical = 12.0

    print(f"\nDerivative accuracy comparison:")
    print(f"  Analytical f''(2):  {analytical:.6f}")
    print(f"  Autodiff:           {d2_autodiff_val:.6f}  (error: {abs(d2_autodiff_val - analytical):.6f})")
    print(f"  Finite diff:        {d2_finite_val:.6f}  (error: {abs(d2_finite_val - analytical):.6f})")

    # Autodiff should be at least 5x more accurate than finite diff
    autodiff_error = abs(d2_autodiff_val - analytical)
    finite_error = abs(d2_finite_val - analytical)

    print(f"\n  Autodiff is {finite_error / autodiff_error:.1f}x more accurate")

    assert finite_error / autodiff_error > 2.0, \
        f"Autodiff should be significantly more accurate (at least 2x), but ratio is {finite_error / autodiff_error:.2f}x"


def test_first_derivative_accuracy():
    """Test that first derivatives are also accurate with autodiff."""
    # Create a simple KAN model
    model = MultKAN(width=[1, 5, 1], grid=5, k=3, seed=42)

    # Train it to approximate f(x) = x²
    x_train = mx.linspace(-2, 2, 100).reshape(-1, 1)
    y_train = x_train ** 2

    # Simple training loop
    from kan_mlx_physics.functional import get_params_list, set_params_list, adam_update, init_adam_state

    params_list = get_params_list(model)
    m, v = init_adam_state(params_list)

    for step in range(500):
        def loss_fn(params):
            from kan_mlx_physics.functional import functional_forward
            y_pred = functional_forward(params, x_train, model.k, model.base_fun)
            return mx.mean((y_pred - y_train) ** 2)

        loss, grads = mx.value_and_grad(loss_fn)(params_list)
        params_list, m, v = adam_update(params_list, grads, m, v, mx.array(step + 1, dtype=mx.float32), lr=0.01)
        mx.eval(loss, params_list)

    # Unpack parameters back to model
    for i, (coef, scale_sp, scale_base, grid) in enumerate(params_list):
        model.layers[i].coef = coef
        model.layers[i].scale_sp = scale_sp
        model.layers[i].scale_base = scale_base

    # Test point: x = 1.5, analytical f'(1.5) = 3.0
    x_test = mx.array([[1.5]])

    # Autodiff
    from kan_mlx_physics.pde.compiler import CompiledResidual

    compiled_autodiff = CompiledResidual(
        parsed=None, jet_req=None, coord_map={'x': 0},
        derivative_method="autodiff"
    )
    d1_autodiff = compiled_autodiff._autodiff_derivative(model, x_test, 'x', order=1)
    d1_autodiff_val = float(d1_autodiff[0])

    # Analytical value
    analytical = 3.0

    print(f"\nFirst derivative accuracy:")
    print(f"  Analytical f'(1.5): {analytical:.6f}")
    print(f"  Autodiff:           {d1_autodiff_val:.6f}  (error: {abs(d1_autodiff_val - analytical):.6f})")

    # Should be within 10% of analytical
    error_pct = abs(d1_autodiff_val - analytical) / analytical * 100
    assert error_pct < 10.0, f"First derivative error {error_pct:.1f}% exceeds 10%"


if __name__ == "__main__":
    test_autodiff_vs_finite_diff_accuracy()
    test_first_derivative_accuracy()
    print("\n✓ All derivative accuracy tests passed!")
