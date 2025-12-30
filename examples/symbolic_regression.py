"""Symbolic Regression Example for KAN-MLX-Physics.

This example demonstrates how KANs can discover symbolic formulas:
1. Train a KAN on synthetic data from f(x,y) = sin(pi*x) + y^2
2. Use auto_symbolic to discover the underlying formula
3. Extract and display the formula in both text and LaTeX formats
"""

import numpy as np
from kan_mlx_physics import MultKAN, create_dataset, list_symbolic


def main():
    # Show available symbolic functions
    print("Available symbolic functions:")
    print(list_symbolic())
    print()

    # Define target function: f(x, y) = sin(pi * x) + y^2
    def target_fn(x):
        return np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2

    # Create dataset
    print("=" * 50)
    print("Creating dataset for f(x,y) = sin(πx) + y²")
    print("=" * 50)
    dataset = create_dataset(
        f=target_fn,
        n_var=2,
        ranges=(-1, 1),
        train_num=1000,
        test_num=200,
        seed=42,
    )

    # Create a simple KAN: 2 inputs -> 1 output (single layer)
    # This architecture matches the structure of the target function
    print("\nCreating KAN model with width=[2, 1]...")
    model = MultKAN(
        width=[2, 1],  # Direct connection: each input has its own activation to output
        grid=10,       # More grid points for better spline fitting
        k=3,
        seed=42,
    )

    print(model.summary())

    # Train the model
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)
    history = model.fit(
        dataset,
        opt="Adam",
        steps=200,
        lr=0.02,
        lamb=0.001,
        update_grid=True,
        grid_update_freq=20,
        stop_grid_update_step=100,
        log_freq=20,
        verbose=True,
    )

    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final test loss: {history['test_loss'][-1]:.6f}")

    # Suggest symbolic functions for each edge
    print("\n" + "=" * 50)
    print("Suggesting symbolic functions for each edge...")
    print("=" * 50)

    for i in range(model.width[0]):
        print(f"\nEdge (0, {i}, 0) - from input x_{i} to output:")
        suggestions = model.suggest_symbolic(0, i, 0, top_k=5)
        for fn_name, r2, params in suggestions:
            print(f"  {fn_name:12s} R² = {r2:.4f}")

    # Fix symbolic functions based on suggestions
    print("\n" + "=" * 50)
    print("Fixing symbolic functions...")
    print("=" * 50)

    # Edge (0,0,0): should be sin (for sin(pi*x))
    model.fix_symbolic(0, 0, 0, "sin", verbose=True)

    # Edge (0,1,0): should be x^2 (for y^2)
    model.fix_symbolic(0, 1, 0, "x^2", verbose=True)

    # Extract symbolic formula
    print("\n" + "=" * 50)
    print("Extracted Symbolic Formula:")
    print("=" * 50)

    formula = model.symbolic_formula(var_names=["x", "y"], verbose=True)

    # LaTeX formula
    print("\n" + "=" * 50)
    print("LaTeX Formula:")
    print("=" * 50)
    latex = model.symbolic_formula_latex(var_names=["x", "y"])
    print(f"\n  ${latex}$\n")

    # Typst formula
    print("=" * 50)
    print("Typst Formula:")
    print("=" * 50)
    typst = model.symbolic_formula_typst(var_names=["x", "y"])
    print(f"\n  ${typst}$\n")

    # Save visualizations
    print("\n" + "=" * 50)
    print("Saving visualizations...")
    print("=" * 50)

    model.plot_kan(
        folder="./figures",
        title="KAN: sin(πx) + y²",
        in_vars=["x", "y"],
        out_vars=["f"],
    )
    print("  Saved: ./figures/kan_network.png")

    model.plot_activations(0, folder="./figures")
    print("  Saved: ./figures/activations_layer_0.png")

    # Also try auto_symbolic on a fresh model
    print("\n" + "=" * 50)
    print("Testing auto_symbolic on fresh model...")
    print("=" * 50)

    model2 = MultKAN(width=[2, 1], grid=10, k=3, seed=123)
    model2.fit(dataset, steps=200, lr=0.02, lamb=0.001, verbose=False)

    fixed = model2.auto_symbolic(r2_threshold=0.95, verbose=True)

    print("\nAuto-detected formula:")
    model2.symbolic_formula(var_names=["x", "y"])

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
