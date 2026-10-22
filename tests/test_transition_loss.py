"""Tests for TransitionLoss — the coupled left/right ⋆-genvalue residual.

A non-diagonal (transition) Wigner distribution must satisfy H⋆W=E_L W AND W⋆H=E_R W
with two independent eigenvalues. TransitionLoss compiles both equation strings and sums
their squared residuals; paired with trainable_params("E_L","E_R") it trains both.
"""
import mlx.core as mx
import numpy as np
from kan_mlx_physics.pde import (
    PDEBuilder, TransitionLoss, NonTrivialLoss, DecayLoss, RegularizationLoss,
)

# Coupled transformed Taub-x pair (differ by cross-term sign and exp(t) sign).
LEFT = ("-(Derivative(w, t, 2) + hbar*Derivative(w, x, t)"
        " + (hbar**2/4)*Derivative(w, x, 2)) + (exp(4*x)/9)*exp(-2*hbar*t)*w = E_L*w")
RIGHT = ("-(Derivative(w, t, 2) - hbar*Derivative(w, x, t)"
         " + (hbar**2/4)*Derivative(w, x, 2)) + (exp(4*x)/9)*exp(+2*hbar*t)*w = E_R*w")


def test_transition_loss_trains_two_eigenvalues():
    b = (PDEBuilder("Derivative(w, x, 2) = 0")   # placeholder; physics is in TransitionLoss
         .coordinates("x", "t")
         .params(hbar=1.0, E_L=1.0, E_R=1.0)
         .trainable_params("E_L", "E_R")
         .param_bounds(E_L=(None, None), E_R=(None, None))
         .domain([[-1.2, 0.4], [-6, 6]])
         .loss(TransitionLoss(LEFT, RIGHT, coords=["x", "t"],
                              params={"hbar": 1.0}, weight=50.0))
         .loss(NonTrivialLoss(10.0))
         .loss(DecayLoss(20.0, threshold_ratio=0.7))
         .loss(RegularizationLoss(0.02, norm="l1")))
    m, h = (b.phase("s", steps=30, lr=1.5e-3, n_points=400, log_freq=99999)
            .model(width=[2, 6, 1], grid=8, k=3,
                   base_fun=lambda z: mx.zeros_like(z), seed=3,
                   basis="fourier", basis_M=8)
            .solve(verbose=False))
    E_L = float(np.array(h.trainable_params["E_L"]))
    E_R = float(np.array(h.trainable_params["E_R"]))
    assert np.isfinite(E_L) and np.isfinite(E_R)


def test_transition_loss_compute_is_finite_nonneg():
    """The loss itself evaluates to a finite, non-negative scalar on a fixture."""
    from kan_mlx_physics import MultKAN
    from kan_mlx_physics.pde.losses import LossContext

    tl = TransitionLoss(LEFT, RIGHT, coords=["x", "t"], params={"hbar": 1.0})
    model = MultKAN(width=[2, 5, 1], grid=6, k=3,
                    base_fun=lambda z: mx.zeros_like(z), seed=0)
    x = mx.array(np.random.RandomState(0).uniform(
        [-1.2, -6], [0.4, 6], size=(40, 2)).astype(np.float32))
    ctx = LossContext(model=model, x_interior=x,
                      params={"hbar": 1.0, "E_L": 1.0, "E_R": 1.0},
                      trainable_params={"E_L": mx.array([1.0]), "E_R": mx.array([1.0])})
    val = float(tl.compute(ctx))
    assert np.isfinite(val) and val >= 0.0
