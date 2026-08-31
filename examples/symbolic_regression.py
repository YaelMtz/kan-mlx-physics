"""Symbolic Regression Example for KAN-MLX-Physics.

This example demonstrates how KANs can discover symbolic formulas:
1. Train a KAN on synthetic data from f(x,y) = sin(pi*x) + y^2
2. Use auto_symbolic to discover the underlying formula
3. Extract and display the formula in both text and LaTeX formats

Expected Output:
===============

Available symbolic functions:
['x', '0', '1', 'x^2', 'x^3', 'x^4', 'x^0.5', 'x^-1', 'x^-2', 'sin', 'cos',
 'tan', 'arcsin', 'arccos', 'arctan', 'exp', 'log', 'sinh', 'cosh', 'tanh',
 'abs', 'sign', 'gaussian', 'sigmoid', 'relu', 'softplus']

==================================================
Creating dataset for f(x,y) = sin(πx) + y²
==================================================

Creating KAN model with width=[2, 1]...
MultKAN Summary
========================================
Width: [2, 1]
Depth: 1
Grid: 10
Spline order: 3
Total parameters: 30

Layers:
  [0] KANLayer: 2 → 1 (30 params)

==================================================
Training...
==================================================
Step    0 | Train: 0.296031 | Test: 0.280825 | Reg: 0.002535
Step   20 | Train: 0.009377 | Test: 0.008668 | Reg: 0.002758
Step   40 | Train: 0.000783 | Test: 0.000758 | Reg: 0.002741
Step   60 | Train: 0.000192 | Test: 0.000196 | Reg: 0.002749
Step   80 | Train: 0.000081 | Test: 0.000073 | Reg: 0.002740
Step  100 | Train: 0.000049 | Test: 0.000043 | Reg: 0.002739
Step  199 | Train: 0.000036 | Test: 0.000032 | Reg: 0.002733

Final train loss: 0.000036
Final test loss: 0.000032

==================================================
Suggesting symbolic functions for each edge...
==================================================

Edge (0, 0, 0) - from input x_0 to output:
  sin          R² = 0.9573  ← Correctly identifies sin function
  cos          R² = 0.9566
  gaussian     R² = 0.6740

Edge (0, 1, 0) - from input x_1 to output:
  x^2          R² = 0.9998  ← Correctly identifies quadratic
  cos          R² = 0.9996
  sin          R² = 0.9996

==================================================
Fixing symbolic functions...
==================================================
Edge (0,0,0): sin
  Params: a=-4.141, b=-3.333, c=0.766, d=0.161
  R² = 0.9573
Edge (0,1,0): x^2
  Params: a=-6.566, b=0.303, c=0.039, d=-0.440
  R² = 0.9998

==================================================
Extracted Symbolic Formula:
==================================================

Layer 0 outputs:
  y_0 = (0.77*sin((-4.14*x - 3.33)) + 0.16 + 0.04*((-6.57*y + 0.30))^2 - 0.44)

Final formula:
  f(x, y) = (0.77*sin((-4.14*x - 3.33)) + 0.16 + 0.04*((-6.57*y + 0.30))^2 - 0.44)

LaTeX: $f(x, y) = 0.77 \\sin((-4.14 x - 3.33)) + 0.16 + 0.04 (-6.57 y + 0.30)^2 - 0.44$

==================================================
Auto-symbolic Detection (fresh model):
==================================================
Layer 0:
  (0,0): sin (R²=0.9530)
  (1,0): x^2 (R²=0.9998)

Fixed 2 edges to symbolic functions

Performance Notes:
------------------
- Training converges to ~3.6e-5 loss within 200 steps (~2 seconds on Apple Silicon)
- Symbolic regression correctly identifies both sin and x^2 components
- R² scores > 0.95 indicate excellent symbolic fit
- Auto-symbolic can automatically detect functions without manual fixing
- The model learns affine transformations (a,b,c,d parameters) around the core symbolic functions

Key Features Demonstrated:
--------------------------
1. suggest_symbolic(): Ranks symbolic functions by R² score for each edge
2. fix_symbolic(): Manually sets an edge to a specific symbolic function
3. auto_symbolic(): Automatically detects and fixes symbolic functions above R² threshold
4. symbolic_formula(): Extracts human-readable formula in text format
5. symbolic_formula_latex(): Exports formula for LaTeX documents
6. symbolic_formula_typst(): Exports formula for Typst documents
7. Visualizations saved to ./figures/

The extracted formulas show the learned affine parameters around the core symbolic functions.
These can be further refined using global parameter optimization if needed.
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
