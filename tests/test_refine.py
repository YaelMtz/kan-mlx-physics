"""Tests for the refine() operation."""

import numpy as np
import mlx.core as mx
from kan_mlx_physics import MultKAN, create_dataset


def test_refine_updates_layers():
    """Test that refine() actually updates the model layers."""
    # Create a simple model
    model = MultKAN(width=[2, 3, 1], grid=5, k=3, seed=42)

    initial_grid = model.grid
    initial_layer_grids = [layer.grid.shape[0] for layer in model.layers]

    # Refine to a higher grid
    new_grid = 10
    model.refine(new_grid)

    # Check that model.grid was updated
    assert model.grid == new_grid, f"Expected grid={new_grid}, got {model.grid}"

    # Check that all layers have the new grid
    for i, layer in enumerate(model.layers):
        # Grid shape is (in_dim, num_grid_points)
        # num_grid_points = num_grid + 2*k + 1 for extended grid with boundaries
        actual_grid_size = layer.grid.shape[1]
        expected_grid_size = new_grid + 2 * layer.k + 1  # Extended grid formula
        assert actual_grid_size == expected_grid_size, \
            f"Layer {i}: expected grid size {expected_grid_size}, got {actual_grid_size}"

    print("✓ refine() correctly updates all layers")


def test_refine_preserves_function():
    """Test that refine() preserves the learned function."""
    # Create a simple dataset
    def target_fn(x):
        return np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2

    dataset = create_dataset(
        f=target_fn,
        n_var=2,
        ranges=(-1, 1),
        train_num=500,
        test_num=100,
        seed=42,
    )

    # Train a model
    model = MultKAN(width=[2, 5, 1], grid=5, k=3, seed=42)
    model.fit(
        dataset,
        opt="Adam",
        steps=50,
        lr=0.01,
        batch_size=-1,
        verbose=False,
    )

    # Evaluate before refinement
    x_test = dataset['test_input']
    y_before = model(x_test)

    # Refine the grid
    model.refine(new_grid=10)

    # Evaluate after refinement
    y_after = model(x_test)

    # Check that outputs are similar (refinement shouldn't drastically change the function)
    diff = mx.mean(mx.abs(y_before - y_after))
    print(f"Mean absolute difference after refine: {float(diff):.6f}")

    # Should be relatively small since we're just changing grid resolution
    assert float(diff) < 0.5, f"Function changed too much after refine: {float(diff)}"

    print("✓ refine() preserves learned function (within tolerance)")


def test_refine_allows_continued_training():
    """Test that model can continue training after refinement."""
    # Create a dataset
    def target_fn(x):
        return np.sin(np.pi * x[:, 0])

    dataset = create_dataset(
        f=target_fn,
        n_var=1,
        ranges=(-1, 1),
        train_num=200,
        test_num=50,
        seed=42,
    )

    # Train initially
    model = MultKAN(width=[1, 3, 1], grid=3, k=3, seed=42)
    history1 = model.fit(
        dataset,
        opt="Adam",
        steps=30,
        lr=0.01,
        batch_size=-1,
        verbose=False,
    )

    loss_before_refine = history1['train_loss'][-1]

    # Refine
    model.refine(new_grid=7)

    # Continue training
    history2 = model.fit(
        dataset,
        opt="Adam",
        steps=30,
        lr=0.01,
        batch_size=-1,
        verbose=False,
    )

    loss_after_refine = history2['train_loss'][-1]

    print(f"Loss before refine: {loss_before_refine:.6f}")
    print(f"Loss after refine + training: {loss_after_refine:.6f}")

    # Training should still work and potentially improve
    # (or at least not crash)
    assert loss_after_refine < 1.0, "Model should still be able to train after refine"

    print("✓ Model can continue training after refine()")


def test_refine_with_symbolic_layers():
    """Test that refine() works when model has symbolic layers."""
    # Create and train a model
    def target_fn(x):
        return np.sin(x[:, 0])

    dataset = create_dataset(
        f=target_fn,
        n_var=1,
        ranges=(-np.pi, np.pi),
        train_num=200,
        test_num=50,
        seed=42,
    )

    model = MultKAN(width=[1, 1], grid=5, k=3, seed=42)
    model.fit(dataset, opt="Adam", steps=50, lr=0.01, batch_size=-1, verbose=False)

    # Auto-symbolify
    model.auto_symbolic()

    # Check that we have symbolic functions
    has_symbolic = any(
        model.symbolic_funs[0].fns_name[i][j] != ""
        for i in range(model.symbolic_funs[0].in_dim)
        for j in range(model.symbolic_funs[0].out_dim)
    )

    if has_symbolic:
        print("Model has symbolic functions before refine")

    # Refine should work even with symbolic layers
    try:
        model.refine(new_grid=10)
        print("✓ refine() works with symbolic layers present")
    except Exception as e:
        raise AssertionError(f"refine() failed with symbolic layers: {e}")


if __name__ == "__main__":
    print("Running refine() tests...")
    print("=" * 50)

    test_refine_updates_layers()
    print()

    test_refine_preserves_function()
    print()

    test_refine_allows_continued_training()
    print()

    test_refine_with_symbolic_layers()
    print()

    print("=" * 50)
    print("All tests passed! ✓")
