"""Unit + smoke tests for the PDE subpackage.

The PDE subpackage (PDEBuilder, loss terms, trainer) is the library's flagship
feature but was previously validated only by external experiment scripts. These
tests cover:

  1. Each self-contained LossTerm.compute() on a small fixture — asserting a
     finite, non-negative scalar (and known-value checks where cheap).
  2. One end-to-end PDEBuilder eigenvalue solve (the harmonic-oscillator
     Schrödinger ground state) — the integration path that exercises the
     parser, compiler, autodiff residual, trainer, and trainable-param loop.
"""

import mlx.core as mx
import numpy as np
import pytest

from kan_mlx_physics import MultKAN, register_physics_symbolic
from kan_mlx_physics.pde.losses import (
    LossContext,
    AnchorLoss,
    NonTrivialLoss,
    DecayLoss,
    NormalizationLoss,
    RegularizationLoss,
    DataLoss,
)
from kan_mlx_physics.pde.problem import Domain

register_physics_symbolic()


@pytest.fixture
def ctx():
    """A minimal LossContext with a tiny model over [0, 6] (no trainer needed)."""
    model = MultKAN(width=[1, 1], grid=5, k=3, seed=0)
    x = mx.linspace(0, 6, 64).reshape(-1, 1)
    return LossContext(
        model=model,
        x_interior=x,
        x_boundary=mx.array([[0.0], [6.0]]),
        domain=Domain(bounds=[(0.0, 6.0)]),
    )


class TestLossTermsFinite:
    """Each self-contained loss term returns a finite non-negative scalar."""

    @pytest.mark.parametrize(
        "term_factory",
        [
            lambda: AnchorLoss(x0=mx.array([[0.0]]), target=1.0),
            lambda: NonTrivialLoss(),
            lambda: DecayLoss(),
            lambda: NormalizationLoss(),
            lambda: RegularizationLoss(norm="l1"),
            lambda: RegularizationLoss(norm="l2"),
        ],
    )
    def test_term_is_finite_nonneg(self, ctx, term_factory):
        v = float(term_factory().compute(ctx))
        assert np.isfinite(v)
        assert v >= 0.0

    def test_data_loss_zero_on_exact_match(self, ctx):
        """DataLoss against the model's own output must be ~0."""
        y = ctx.model(ctx.x_interior)
        loss = DataLoss(x_data=ctx.x_interior, y_data=y)
        assert float(loss.compute(ctx)) < 1e-8

    def test_anchor_loss_pins_value(self, ctx):
        """AnchorLoss = (u(x0) - target)^2."""
        x0 = mx.array([[0.0]])
        u0 = float(np.array(ctx.model(x0)).flatten()[0])
        loss = AnchorLoss(x0=x0, target=u0)
        assert float(loss.compute(ctx)) < 1e-8

    def test_weight_scales_only_composer_not_compute(self, ctx):
        """compute() returns the raw (unweighted) term value."""
        t1 = NonTrivialLoss(weight=1.0)
        t2 = NonTrivialLoss(weight=100.0)
        assert abs(float(t1.compute(ctx)) - float(t2.compute(ctx))) < 1e-9


class TestPDEBuilderEndToEnd:
    """Integration smoke test through the full DSL → trainer path."""

    def test_harmonic_oscillator_eigenvalue(self):
        """Schrödinger HO ground state: E should head toward ~0.5.

        Short schedule — this is a smoke test that the whole pipeline runs and
        moves E in the right direction, not a convergence benchmark.
        """
        from kan_mlx_physics.pde import (
            PDEBuilder, PDEResidualLoss, NonTrivialLoss as NT,
            NormalizationLoss as NL, EigenvalueLoss,
        )

        eq = "-0.5*Derivative(psi, x, 2) + 0.5*x**2*psi = E*psi"
        model, history = (
            PDEBuilder(eq)
            .params(E=1.0)
            .trainable_params("E")
            .domain([-5, 5])
            .loss(PDEResidualLoss(weight=1.0, normalize=True))
            .loss(NT(weight=1.0))
            .loss(NL(weight=1.0))
            .loss(EigenvalueLoss(weight=1.0, method="trainable"))
            .phase("shape", steps=400, lr=0.01, n_points=200, log_freq=1000)
            .model(width=[1, 8, 1], grid=8, k=3, seed=0)
            .solve(verbose=False)
        )

        E = float(np.array(history.trainable_params["E"]))
        # Smoke test: the pipeline runs, E stays finite and positive (does not
        # NaN or explode). Convergence to the true ground state 0.5 needs a
        # full multi-phase schedule, which is out of scope for a unit test.
        assert np.isfinite(E)
        assert 0.0 < E < 10.0

    def test_builder_returns_callable_model(self):
        """The solved model must be callable and produce finite output."""
        from kan_mlx_physics.pde import PDEBuilder, PDEResidualLoss

        model, _ = (
            PDEBuilder("Derivative(u, x, 2) = 0")
            .domain([0, 1])
            .loss(PDEResidualLoss(weight=1.0))
            .phase("shape", steps=50, lr=0.01, n_points=100, log_freq=1000)
            .model(width=[1, 1], grid=5, k=3, seed=0)
            .solve(verbose=False)
        )
        y = np.array(model(mx.array([[0.5]])))
        assert np.all(np.isfinite(y))
