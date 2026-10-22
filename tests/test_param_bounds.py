"""Tests for `.param_bounds(...)` — allowing non-positive trainable eigenvalues.

The trainer historically clamped every trainable parameter to >= 0.01 after each Adam
step (positivity projection). That is correct for an energy above a well bottom but WRONG
for problems whose eigenvalue is genuinely negative (Morse, hydrogen, the Lorentzian
Wheeler-DeWitt / cosmological constraints). `.param_bounds(E=(None, None))` lets such an
eigenvalue train freely, while the default (no bounds) preserves the >= 0.01 floor.
"""
import mlx.core as mx
import numpy as np
from kan_mlx_physics.pde import PDEBuilder, PDEResidualLoss, EigenvalueLoss


def _solve(bounds, e_init, steps=200):
    b = (PDEBuilder("Derivative(w, x, 2) + E*w = 0")
         .params(E=e_init).trainable_params("E")
         .domain([[-3, 3]])
         .loss(PDEResidualLoss(1.0))
         .loss(EigenvalueLoss(1.0, method="trainable")))
    if bounds is not None:
        b = b.param_bounds(E=bounds)
    _, h = (b.phase("s", steps=steps, lr=5e-3, n_points=200, log_freq=99999)
            .model(width=[1, 4, 1], grid=6, k=3,
                   base_fun=lambda z: mx.zeros_like(z), seed=0)
            .solve(verbose=False))
    return float(np.array(h.trainable_params["E"]))


def test_default_still_floors_positive():
    """Backward-compat: with no bounds, a negative init is floored to the 0.01 positivity
    bound, exactly as before."""
    E = _solve(None, -5.0)
    assert E >= 0.0 and abs(E - 0.01) < 0.05


def test_unbounded_allows_negative_eigenvalue():
    """With (None, None) bounds, a negative-initialized eigenvalue stays negative instead
    of being clamped to 0.01."""
    E = _solve((None, None), -5.0)
    assert E < 0.0, f"eigenvalue should remain negative, got {E}"


def test_explicit_lower_bound_is_respected():
    """An explicit finite lower bound clamps as requested."""
    E = _solve((-2.0, None), -5.0)
    assert E >= -2.0 - 1e-3
