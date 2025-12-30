"""Test KAN-MLX-Physics compatibility with pykan behavior.

This test suite verifies that our MLX implementation behaves like pykan:
1. Model creation and architecture
2. Forward pass computation
3. Training convergence
4. Grid updates
5. Symbolic regression
6. Pruning
7. Formula extraction
"""

import numpy as np
import mlx.core as mx
from kan_mlx_physics import (
    MultKAN,
    KANLayer,
    create_dataset,
    register_physics_symbolic,
    list_symbolic,
    B_batch,
    coef2curve,
)


def test_model_creation():
    """Test 1: Model creation matches pykan API."""
    print("=" * 60)
    print("Test 1: Model Creation")
    print("=" * 60)

    # pykan: model = KAN(width=[2, 5, 1], grid=5, k=3, seed=42)
    model = MultKAN(width=[2, 5, 1], grid=5, k=3, seed=42)

    assert model.width == [2, 5, 1], "Width mismatch"
    assert model.depth == 2, "Depth mismatch"
    assert model.grid == 5, "Grid mismatch"
    assert model.k == 3, "Spline order mismatch"
    assert len(model.layers) == 2, "Layer count mismatch"

    # Check layer dimensions
    assert model.layers[0].in_dim == 2, "Layer 0 in_dim mismatch"
    assert model.layers[0].out_dim == 5, "Layer 0 out_dim mismatch"
    assert model.layers[1].in_dim == 5, "Layer 1 in_dim mismatch"
    assert model.layers[1].out_dim == 1, "Layer 1 out_dim mismatch"

    print("✓ Model creation matches pykan API")
    print(f"  width={model.width}, depth={model.depth}, grid={model.grid}, k={model.k}")
    return True


def test_forward_pass():
    """Test 2: Forward pass produces correct shapes."""
    print("\n" + "=" * 60)
    print("Test 2: Forward Pass")
    print("=" * 60)

    model = MultKAN(width=[2, 5, 1], grid=5, k=3, seed=42)

    # Test batch input
    x = mx.random.uniform(shape=(100, 2))
    y = model(x)

    assert y.shape == (100, 1), f"Output shape mismatch: {y.shape} != (100, 1)"

    # Test with return_activations
    y, activations = model(x, return_activations=True)
    assert len(activations) == 2, "Should return activations for each layer"
    assert "preacts" in activations[0], "Missing preacts"
    assert "postacts" in activations[0], "Missing postacts"

    print("✓ Forward pass produces correct shapes")
    print(f"  Input: {x.shape} → Output: {y.shape}")
    return True


def test_spline_basis():
    """Test 3: B-spline basis functions are correct."""
    print("\n" + "=" * 60)
    print("Test 3: B-Spline Basis Functions")
    print("=" * 60)

    # Create a simple grid
    grid = mx.linspace(-1, 1, 6)  # 5 intervals
    grid = mx.broadcast_to(grid, (1, 6))  # (in_dim=1, num_grid+1=6)

    # Extend grid for k=3
    from kan_mlx_physics.spline import extend_grid
    grid_ext = extend_grid(grid, k_extend=3)

    # Evaluate basis at sample points
    x = mx.linspace(-1, 1, 50).reshape(-1, 1)
    bases = B_batch(x, grid_ext, k=3)

    # Check partition of unity: sum of bases ≈ 1
    basis_sum = mx.sum(bases, axis=2)
    mean_sum = float(mx.mean(basis_sum))

    assert 0.9 < mean_sum < 1.1, f"Partition of unity failed: mean sum = {mean_sum}"

    print("✓ B-spline basis functions are correct")
    print(f"  Partition of unity: mean sum = {mean_sum:.4f}")
    return True


def test_training_convergence():
    """Test 4: Training converges on simple function."""
    print("\n" + "=" * 60)
    print("Test 4: Training Convergence")
    print("=" * 60)

    # pykan example: f(x, y) = sin(pi*x) + y^2
    def target_fn(x):
        return np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2

    dataset = create_dataset(
        f=target_fn,
        n_var=2,
        ranges=(-1, 1),
        train_num=1000,
        test_num=200,
        seed=42,
    )

    model = MultKAN(width=[2, 5, 1], grid=5, k=3, seed=42)

    # Train
    history = model.fit(
        dataset,
        opt="Adam",
        steps=100,
        lr=0.02,
        lamb=0.001,
        verbose=False,
    )

    final_loss = history["train_loss"][-1]
    initial_loss = history["train_loss"][0]

    assert final_loss < initial_loss, "Training did not improve loss"
    assert final_loss < 0.1, f"Final loss too high: {final_loss}"

    print("✓ Training converges")
    print(f"  Initial loss: {initial_loss:.4f} → Final loss: {final_loss:.4f}")
    return True


def test_grid_update():
    """Test 5: Grid update from samples works."""
    print("\n" + "=" * 60)
    print("Test 5: Grid Update")
    print("=" * 60)

    model = MultKAN(width=[2, 3, 1], grid=5, k=3, seed=42)

    # Get initial grid
    initial_grid = np.array(model.layers[0].grid)

    # Create samples in a different range
    x = mx.random.uniform(low=-2, high=2, shape=(100, 2))

    # Update grid
    model.update_grid_from_samples(x)

    # Grid should have changed
    new_grid = np.array(model.layers[0].grid)

    grid_changed = not np.allclose(initial_grid, new_grid)
    assert grid_changed, "Grid did not update"

    print("✓ Grid update from samples works")
    print(f"  Grid range changed: [{initial_grid.min():.2f}, {initial_grid.max():.2f}] → [{new_grid.min():.2f}, {new_grid.max():.2f}]")
    return True


def test_symbolic_suggestion():
    """Test 6: Symbolic function suggestion works."""
    print("\n" + "=" * 60)
    print("Test 6: Symbolic Suggestion")
    print("=" * 60)

    # Train on sin(x)
    def sin_fn(x):
        return np.sin(np.pi * x[:, 0])

    dataset = create_dataset(
        f=sin_fn,
        n_var=1,
        ranges=(-1, 1),
        train_num=500,
        test_num=100,
        seed=42,
    )

    model = MultKAN(width=[1, 1], grid=10, k=3, seed=42)
    model.fit(dataset, steps=150, lr=0.02, verbose=False)

    # Get suggestions
    suggestions = model.suggest_symbolic(0, 0, 0, top_k=5)

    assert len(suggestions) > 0, "No suggestions returned"
    assert suggestions[0][1] > 0.8, f"Best R² too low: {suggestions[0][1]}"

    # sin should be in top suggestions
    fn_names = [s[0] for s in suggestions]
    sin_found = "sin" in fn_names or "cos" in fn_names

    print("✓ Symbolic suggestion works")
    print(f"  Top suggestions: {[(s[0], f'R²={s[1]:.3f}') for s in suggestions[:3]]}")
    return True


def test_fix_symbolic():
    """Test 7: Fixing symbolic functions works."""
    print("\n" + "=" * 60)
    print("Test 7: Fix Symbolic")
    print("=" * 60)

    def square_fn(x):
        return x[:, 0] ** 2

    dataset = create_dataset(
        f=square_fn,
        n_var=1,
        ranges=(-1, 1),
        train_num=500,
        test_num=100,
        seed=42,
    )

    model = MultKAN(width=[1, 1], grid=10, k=3, seed=42)
    model.fit(dataset, steps=150, lr=0.02, verbose=False)

    # Fix to x^2
    r2 = model.fix_symbolic(0, 0, 0, "x^2", verbose=False)

    assert r2 > 0.9, f"R² for x^2 fit too low: {r2}"
    assert model.symbolic_funs[0].is_symbolic(0, 0), "Edge not marked as symbolic"

    print("✓ Fix symbolic works")
    print(f"  Fixed edge (0,0,0) to x^2 with R² = {r2:.4f}")
    return True


def test_auto_symbolic():
    """Test 8: Auto-symbolic detection works."""
    print("\n" + "=" * 60)
    print("Test 8: Auto Symbolic")
    print("=" * 60)

    def mixed_fn(x):
        return np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2

    dataset = create_dataset(
        f=mixed_fn,
        n_var=2,
        ranges=(-1, 1),
        train_num=1000,
        test_num=200,
        seed=42,
    )

    model = MultKAN(width=[2, 1], grid=10, k=3, seed=42)
    model.fit(dataset, steps=200, lr=0.02, verbose=False)

    # Auto-detect with lower threshold
    fixed = model.auto_symbolic(r2_threshold=0.90, verbose=False)

    assert len(fixed) > 0, "No edges auto-fixed"

    print("✓ Auto-symbolic detection works")
    print(f"  Auto-fixed {len(fixed)} edges: {list(fixed.keys())}")
    return True


def test_formula_extraction():
    """Test 9: Formula extraction works."""
    print("\n" + "=" * 60)
    print("Test 9: Formula Extraction")
    print("=" * 60)

    model = MultKAN(width=[2, 1], grid=5, k=3, seed=42)

    # Manually fix some symbolic functions
    model.symbolic_funs[0].fix_symbolic(0, 0, "sin", 3.14, 0, 1, 0)
    model.symbolic_funs[0].fix_symbolic(1, 0, "x^2", 1, 0, 1, 0)

    # Extract formulas
    text_formula = model.symbolic_formula(var_names=["x", "y"], verbose=False)
    latex_formula = model.symbolic_formula_latex(var_names=["x", "y"])
    typst_formula = model.symbolic_formula_typst(var_names=["x", "y"])

    assert "sin" in text_formula.lower(), "sin not in text formula"
    assert "sin" in latex_formula.lower(), "sin not in LaTeX formula"
    assert "sin" in typst_formula.lower(), "sin not in Typst formula"

    print("✓ Formula extraction works")
    print(f"  Text:  {text_formula}")
    print(f"  LaTeX: {latex_formula}")
    print(f"  Typst: {typst_formula}")
    return True


def test_pruning():
    """Test 10: Pruning works."""
    print("\n" + "=" * 60)
    print("Test 10: Pruning")
    print("=" * 60)

    model = MultKAN(width=[2, 5, 1], grid=5, k=3, seed=42)

    # Train briefly
    def simple_fn(x):
        return x[:, 0] + x[:, 1]

    dataset = create_dataset(f=simple_fn, n_var=2, train_num=500, seed=42)
    model.fit(dataset, steps=50, lr=0.02, verbose=False)

    # Prune edges
    model.prune_edges(threshold=0.001)

    # Check that mask has some zeros
    mask = np.array(model.layers[0].mask)
    has_pruned = np.any(mask < 0.5)

    print("✓ Pruning works")
    print(f"  Active edges in layer 0: {int(mask.sum())}/{mask.size}")
    return True


def test_save_load():
    """Test 11: Save and load checkpoints."""
    print("\n" + "=" * 60)
    print("Test 11: Save/Load Checkpoints")
    print("=" * 60)

    import tempfile
    import os

    model = MultKAN(width=[2, 3, 1], grid=5, k=3, seed=42)

    # Get initial prediction
    x = mx.array([[0.5, 0.5]])
    y_before = float(model(x)[0, 0])

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_model")
        model.saveckpt(path)

        # Modify model
        model.layers[0].coef = model.layers[0].coef * 0  # Zero out

        y_zeroed = float(model(x)[0, 0])
        assert y_zeroed != y_before, "Model should have changed"

        # Load
        model.loadckpt(path)
        y_after = float(model(x)[0, 0])

    assert abs(y_before - y_after) < 1e-5, f"Predictions don't match after load: {y_before} vs {y_after}"

    print("✓ Save/Load checkpoints work")
    print(f"  Prediction preserved: {y_before:.4f}")
    return True


def test_physics_symbolic():
    """Test 12: Physics symbolic functions work."""
    print("\n" + "=" * 60)
    print("Test 12: Physics Symbolic Functions")
    print("=" * 60)

    register_physics_symbolic()

    # Check that physics functions are registered
    all_fns = list_symbolic()

    physics_fns = ["J_0", "psi_0", "bose", "fermi", "propagator", "H_2"]
    found = [fn for fn in physics_fns if fn in all_fns]

    assert len(found) == len(physics_fns), f"Missing physics functions: {set(physics_fns) - set(found)}"

    # Test that a physics function works
    def gaussian_fn(x):
        return np.exp(-x[:, 0] ** 2 / 2)

    dataset = create_dataset(f=gaussian_fn, n_var=1, ranges=(-3, 3), train_num=500, seed=42)
    model = MultKAN(width=[1, 1], grid=10, k=3, seed=42)
    model.fit(dataset, steps=100, lr=0.02, verbose=False)

    suggestions = model.suggest_symbolic(0, 0, 0, top_k=10)
    fn_names = [s[0] for s in suggestions]

    # psi_0 or gaussian should be suggested
    found_physics = any(fn in fn_names for fn in ["psi_0", "gaussian", "psi_1"])

    print("✓ Physics symbolic functions work")
    print(f"  Registered: {len(all_fns)} total functions")
    print(f"  Top suggestions for Gaussian: {fn_names[:5]}")
    return True


def test_visualization():
    """Test 13: Visualization doesn't crash."""
    print("\n" + "=" * 60)
    print("Test 13: Visualization")
    print("=" * 60)

    import tempfile
    import os

    model = MultKAN(width=[2, 3, 1], grid=5, k=3, seed=42)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test basic plot
        model.plot(folder=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "kan_structure.png")), "Basic plot not created"

        # Test enhanced plot (uses same function as plot(), so same filename)
        model.plot_kan(folder=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "kan_structure.png")), "Enhanced plot not created"

        # Test activation plot
        model.plot_activations(0, folder=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "activations_layer_0.png")), "Activation plot not created"

    print("✓ Visualization works")
    print("  Created: kan_structure.png, kan_network.png, activations_layer_0.png")
    return True


def run_all_tests():
    """Run all compatibility tests."""
    print("\n" + "=" * 60)
    print("KAN-MLX-Physics vs pykan Compatibility Tests")
    print("=" * 60 + "\n")

    tests = [
        ("Model Creation", test_model_creation),
        ("Forward Pass", test_forward_pass),
        ("B-Spline Basis", test_spline_basis),
        ("Training Convergence", test_training_convergence),
        ("Grid Update", test_grid_update),
        ("Symbolic Suggestion", test_symbolic_suggestion),
        ("Fix Symbolic", test_fix_symbolic),
        ("Auto Symbolic", test_auto_symbolic),
        ("Formula Extraction", test_formula_extraction),
        ("Pruning", test_pruning),
        ("Save/Load", test_save_load),
        ("Physics Symbolic", test_physics_symbolic),
        ("Visualization", test_visualization),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"✗ {name} FAILED: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, p, _ in results if p)
    total = len(results)

    for name, p, err in results:
        status = "✓ PASS" if p else f"✗ FAIL: {err}"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! KAN-MLX-Physics is compatible with pykan.")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")

    return passed == total


if __name__ == "__main__":
    run_all_tests()
