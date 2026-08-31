"""
Wigner function of the quantum harmonic oscillator, discovered from the phase-space
PDE alone (no analytic target) with a physics-informed KAN.

This is the flagship example: it solves the radial star-genvalue (Moyal) equation

    (r^2 - 2E) W - (hbar^2/4) [4 r^2 W'' + 4 W'] = 0,     hbar = 1,

for the ground state, whose exact solution is W_0(r^2) = e^{-r^2} / pi with
eigenvalue E_0 = 1/2. The network is given ONLY physical constraints — the PDE
residual, the trace normalization Tr(rho) = 1 (i.e. int W dr^2 = 1/pi), decay, a
Dirichlet boundary, and the pure-state purity Tr(rho^2) = 1 — and recovers both
the eigenvalue and the eigenfunction with no analytic answer supplied.

Run:
    uv run python examples/wigner_harmonic_oscillator.py
"""

import mlx.core as mx
import numpy as np

from kan_mlx_physics import register_physics_symbolic
from kan_mlx_physics.pde import (
    PDEBuilder, PDEResidualLoss, NonTrivialLoss, DecayLoss, AnchorLoss,
    EigenvalueLoss, RegularizationLoss, LossTerm,
)

register_physics_symbolic()

DOMAIN = 10.0                      # collocation domain r^2 in [0, DOMAIN]
PURITY_TARGET = 1.0 / (2 * np.pi ** 2)   # int W^2 dr^2 for a pure state (hbar=1)


def W0(r2):
    """Analytic ground-state Wigner function (used only for evaluation)."""
    return np.exp(-r2) / np.pi


class TraceNorm(LossTerm):
    """Tr(rho) = 1  =>  int_0^inf W dr^2 = 1/pi. Data-free."""
    def __init__(self, weight=800.0, **kw):
        super().__init__(weight, **kw)
        self.target = 1.0 / np.pi

    def compute(self, ctx):
        u = ctx.get_u(ctx.x_interior)
        u = mx.squeeze(u, -1) if len(u.shape) > 1 else u
        L = ctx.domain.bounds[0][1] - ctx.domain.bounds[0][0]
        return (mx.mean(u) * L - self.target) ** 2


class Purity(LossTerm):
    """Tr(rho^2) = 1  =>  2 pi^2 int W^2 dr^2 = 1. Data-free pure-state condition."""
    def __init__(self, weight=400.0, **kw):
        super().__init__(weight, **kw)
        self.target = PURITY_TARGET

    def compute(self, ctx):
        u = ctx.get_u(ctx.x_interior)
        u = mx.squeeze(u, -1) if len(u.shape) > 1 else u
        L = ctx.domain.bounds[0][1] - ctx.domain.bounds[0][0]
        return (mx.mean(u ** 2) * L / self.target - 1.0) ** 2


def main():
    equation = ("(r2 - 2*E)*W - (hbar**2/4)*(4*r2*Derivative(W, r2, 2) "
                "+ 4*Derivative(W, r2)) = 0")

    print("Discovering the HO ground-state Wigner function from the PDE alone...")
    model, history = (
        PDEBuilder(equation)
        .params(hbar=1.0, E=0.5)              # E initialised at the target sector
        .trainable_params("E")                # ...but E is trained, not fixed
        .domain([0, DOMAIN])
        .loss(PDEResidualLoss(weight=50.0, normalize=True))
        .loss(NonTrivialLoss(weight=10.0))
        .loss(DecayLoss(weight=50.0, threshold_ratio=0.7))
        .loss(AnchorLoss(x0=mx.array([[DOMAIN]]), target=0.0, weight=500.0, id="bc"))
        .loss(TraceNorm(800.0))
        .loss(Purity(400.0))                  # the data-free purity constraint
        .loss(EigenvalueLoss(weight=3.0, method="trainable"))
        .loss(RegularizationLoss(weight=0.02, norm="l1"))
        .phase("shape",  steps=6000, lr=1.5e-3, n_points=3000,
               use_gradnorm=True, gradnorm_alpha=1.5)
        .phase("refine", steps=4000, lr=6e-4,  n_points=3000)
        # native weighted-Laguerre single edge: W_0 = e^{-s} is one basis element
        .model(width=[1, 1], grid=28, k=3, basis="laguerre",
               basis_kwargs={"alpha": 0.0, "weighted": True,
                             "nonnegative_input": True, "fixed_scale": 2.0},
               base_fun=lambda x: mx.zeros_like(x), noise_scale=0.05,
               grid_range=(0, DOMAIN), seed=3)
        .solve(verbose=False)
    )

    E = float(np.array(history.trainable_params["E"]))

    # Evaluate against the analytic solution (never used in training).
    r2 = np.linspace(0, DOMAIN, 1000)
    W = np.array(model(mx.array(r2.reshape(-1, 1).astype(np.float32)))).flatten()
    Wex = W0(r2)
    dr = r2[1] - r2[0]
    l2 = min(np.sqrt(np.sum((W - Wex) ** 2) * dr),
             np.sqrt(np.sum((W + Wex) ** 2) * dr)) / np.sqrt(np.sum(Wex ** 2) * dr)

    print("\nResult (no analytic target was used in training):")
    print(f"  eigenvalue   E = {E:.4f}   (exact 0.5,  error {abs(E-0.5)/0.5*100:.2f}%)")
    print(f"  peak      W(0) = {W[0]:.4f}   (exact {1/np.pi:.4f})")
    print(f"  trace   int W  = {np.trapezoid(W, r2):.4f}   (exact {1/np.pi:.4f})")
    print(f"  purity int W^2 = {np.trapezoid(W**2, r2):.5f}   (exact {PURITY_TARGET:.5f})")
    print(f"  L2 vs analytic = {l2*100:.1f}%")


if __name__ == "__main__":
    main()
