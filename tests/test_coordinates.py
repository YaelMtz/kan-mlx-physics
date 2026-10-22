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


def test_declared_coords_override_known_coord_params():
    """A parameter whose name happens to be in KNOWN_COORDS (e.g. 'a') must stay a
    PARAMETER when coordinates are declared explicitly — not be miscast as a
    coordinate (which corrupted the coord_map and broke the residual)."""
    eq = ("-(1/(2*m))*(Derivative(w, t, 2) - hbar*Derivative(w, x, t)"
          " + (hbar**2/4)*Derivative(w, x, 2))"
          " + D*(exp(-2*a*x - a*hbar*t) - 2*exp(-a*x - a*hbar*t/2))*w = E*w")
    a = analyze(parse(eq, extra_coords=["x", "t"]))
    assert sorted(a.coordinates) == ["t", "x"]      # only declared coords
    assert "a" not in a.coordinates                 # 'a' is a parameter
    assert "a" in a.parameters


def test_mixed_and_second_derivatives_with_exponential_potential():
    """Regression: Derivative(w,t,2) combined with an exponential term and a mixed
    Derivative(w,x,t) must not leak an (N, dim) shape and crash the residual —
    the full Mellin-transformed Morse equation must build and solve."""
    import mlx.core as mx
    import numpy as np
    from kan_mlx_physics.pde import PDEBuilder, PDEResidualLoss, EigenvalueLoss

    eq = ("-(1/(2*m))*(Derivative(w, t, 2) - hbar*Derivative(w, x, t)"
          " + (hbar**2/4)*Derivative(w, x, 2))"
          " + D*(exp(-2*a*x - a*hbar*t) - 2*exp(-a*x - a*hbar*t/2))*w = E*w")
    m, h = (PDEBuilder(eq)
            .coordinates("x", "t")
            .params(m=1.0, hbar=1.0, D=8.0, a=0.5, E=1.0).trainable_params("E")
            .domain([[-2, 6], [-10, 10]])
            .loss(PDEResidualLoss(weight=1.0))
            .loss(EigenvalueLoss(weight=1.0, method="trainable"))
            .phase("s", steps=3, lr=1e-3, n_points=60, log_freq=99999)
            .model(width=[2, 4, 1], grid=6, k=3,
                   base_fun=lambda z: mx.zeros_like(z), seed=0)
            .solve(verbose=False))
    assert np.isfinite(float(np.array(h.trainable_params["E"])))


def test_mixed_partial_is_computed_correctly():
    """A mixed derivative Derivative(w, x, t) must evaluate to the true ∂²w/∂x∂t, not
    silently collapse to the first-axis derivative. Checked against an analytic model
    w = sin(x) cos(t), whose ∂²w/∂x∂t = -cos(x) sin(t)."""
    import mlx.core as mx
    import numpy as np
    from kan_mlx_physics.pde.compiler import compile_residual

    p = parse("Derivative(w, x, t) = 0", extra_coords=["x", "t"])
    comp = compile_residual(p, analyze(p).jet_req)

    class _Model:
        def __call__(self, X):
            return (mx.sin(X[:, 0]) * mx.cos(X[:, 1])).reshape(-1, 1)

    X = mx.array(np.array([[0.3, 0.5], [1.0, 0.2], [-0.7, 1.1]], dtype=np.float32))
    got = np.array(comp.evaluate(_Model(), X, {}))        # residual == the mixed partial
    want = np.array([-np.cos(0.3) * np.sin(0.5),
                     -np.cos(1.0) * np.sin(0.2),
                     -np.cos(-0.7) * np.sin(1.1)])
    assert np.allclose(got, want, atol=1e-3), f"{got} vs {want}"
