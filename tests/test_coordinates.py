"""Tests for the explicit `.coordinates(...)` declaration API.

The parser recognizes a fixed set of coordinate names; any other bare symbol that
appears only algebraically — notably a momentum like `p` in a phase-space / Wigner
equation — was otherwise silently classified as a *parameter* and then failed at
evaluation (``mx.full(..., None)``). Declaring coordinates fixes this.
"""
from kan_mlx_physics.pde.parser import parse
from kan_mlx_physics.pde.analysis import analyze


EQ_P = "(p**2/2 - E)*W - (hbar**2/8)*Derivative(W, x, 2) = 0"


def test_momentum_symbol_without_declaration_is_not_a_coordinate():
    """Backward-compat: an undeclared `p` is NOT treated as a coordinate."""
    a = analyze(parse(EQ_P))
    assert "p" not in a.coordinates          # old behavior preserved
    assert "x" in a.coordinates


def test_declared_momentum_becomes_a_coordinate():
    """extra_coords promotes `p` to a coordinate and the dimension updates."""
    a = analyze(parse(EQ_P, extra_coords=["x", "p"]))
    assert "p" in a.coordinates and "x" in a.coordinates
    assert a.dimension == 2                   # genuine 2D phase space


def test_existing_equations_unchanged():
    """Regression: standard equations parse with the same coordinates/dimension."""
    a_ho = analyze(parse(
        "(r2 - 2*E)*W - (hbar**2/4)*(4*r2*Derivative(W, r2, 2)"
        " + 4*Derivative(W, r2)) = 0"))
    assert sorted(a_ho.coordinates) == ["r2"] and a_ho.dimension == 1

    a_2d = analyze(parse(
        "-0.5*Derivative(psi, x, 2) - 0.5*Derivative(psi, y, 2)"
        " + 0.5*(x**2 + y**2)*psi = E*psi"))
    assert sorted(a_2d.coordinates) == ["x", "y"] and a_2d.dimension == 2


def test_builder_coordinates_api_runs_end_to_end():
    """The .coordinates(...) builder method lets a p-momentum equation solve."""
    import mlx.core as mx
    import numpy as np
    from kan_mlx_physics.pde import PDEBuilder, PDEResidualLoss, EigenvalueLoss

    m, h = (PDEBuilder(EQ_P)
            .coordinates("x", "p")                 # declare p as a coordinate
            .params(hbar=1.0, E=1.0).trainable_params("E")
            .domain([[0, 1], [-10, 10]])           # bounds in declared order
            .loss(PDEResidualLoss(weight=1.0))
            .loss(EigenvalueLoss(weight=1.0, method="trainable"))
            .phase("s", steps=5, lr=1e-3, n_points=80, log_freq=99999)
            .model(width=[2, 4, 1], grid=6, k=3,
                   base_fun=lambda z: mx.zeros_like(z), seed=0)
            .solve(verbose=False))
    E = float(np.array(h.trainable_params["E"]))
    assert np.isfinite(E)                           # no crash, E stays finite
