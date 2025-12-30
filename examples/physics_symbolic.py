"""Physics Symbolic Regression Example for KAN-MLX-Physics.

This example demonstrates discovering physics formulas using KANs:
1. Harmonic oscillator ground state: ψ₀(x) = exp(-x²/2)
2. Yukawa potential: V(r) = exp(-r)/r
3. Hydrogen radial wavefunction
"""

import numpy as np
from kan_mlx_physics import (
    MultKAN,
    create_dataset,
    list_symbolic,
    register_physics_symbolic,
    list_physics_symbolic,
)


def main():
    # Register physics symbolic functions
    print("Registering physics symbolic functions...")
    register_physics_symbolic()

    # Show all available functions organized by category
    print("\n" + "=" * 60)
    print("Available Physics Symbolic Functions:")
    print("=" * 60)
    physics_fns = list_physics_symbolic()
    for category, fns in physics_fns.items():
        print(f"\n{category}:")
        print(f"  {', '.join(fns)}")

    # Also show base functions
    print(f"\nBase functions: {list_symbolic()[:15]}...")
    print(f"Total functions available: {len(list_symbolic())}")

    # =========================================================================
    # Example 1: Harmonic Oscillator Ground State
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 1: Harmonic Oscillator Ground State")
    print("Target: ψ₀(x) = exp(-x²/2)")
    print("=" * 60)

    def psi_0(x):
        return np.exp(-x[:, 0]**2 / 2)

    dataset1 = create_dataset(
        f=psi_0,
        n_var=1,
        ranges=(-3, 3),
        train_num=500,
        test_num=100,
        seed=42,
    )

    model1 = MultKAN(width=[1, 1], grid=10, k=3, seed=42)
    model1.fit(dataset1, steps=150, lr=0.02, verbose=False)

    print("\nTop symbolic suggestions:")
    suggestions = model1.suggest_symbolic(0, 0, 0, top_k=5)
    for fn, r2, _ in suggestions:
        print(f"  {fn:15s} R² = {r2:.4f}")

    # Fix to psi_0 (harmonic oscillator ground state)
    model1.fix_symbolic(0, 0, 0, "psi_0", verbose=True)

    print("\nFormula (text):")
    model1.symbolic_formula(var_names=["x"], verbose=True)

    print("\nFormula (LaTeX):")
    print(f"  ${model1.symbolic_formula_latex(var_names=['x'])}$")

    # =========================================================================
    # Example 2: Yukawa Potential
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 2: Yukawa Potential")
    print("Target: V(r) = exp(-r)/r")
    print("=" * 60)

    def yukawa_potential(x):
        r = np.abs(x[:, 0]) + 0.1  # Avoid r=0
        return np.exp(-r) / r

    dataset2 = create_dataset(
        f=yukawa_potential,
        n_var=1,
        ranges=(0.1, 5),
        train_num=500,
        test_num=100,
        seed=42,
    )

    model2 = MultKAN(width=[1, 1], grid=15, k=3, seed=42)
    model2.fit(dataset2, steps=200, lr=0.02, verbose=False)

    print("\nTop symbolic suggestions:")
    suggestions = model2.suggest_symbolic(0, 0, 0, top_k=5)
    for fn, r2, _ in suggestions:
        print(f"  {fn:15s} R² = {r2:.4f}")

    # =========================================================================
    # Example 3: Bessel Function J₀
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 3: Bessel Function J₀")
    print("Target: J₀(x)")
    print("=" * 60)

    from scipy import special

    def bessel_j0(x):
        return special.j0(x[:, 0])

    dataset3 = create_dataset(
        f=bessel_j0,
        n_var=1,
        ranges=(0, 10),
        train_num=500,
        test_num=100,
        seed=42,
    )

    model3 = MultKAN(width=[1, 1], grid=20, k=3, seed=42)
    model3.fit(dataset3, steps=200, lr=0.02, verbose=False)

    print("\nTop symbolic suggestions:")
    suggestions = model3.suggest_symbolic(0, 0, 0, top_k=5)
    for fn, r2, _ in suggestions:
        print(f"  {fn:15s} R² = {r2:.4f}")

    # =========================================================================
    # Example 4: Bose-Einstein Distribution
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 4: Bose-Einstein Distribution")
    print("Target: n(ε) = 1/(exp(ε) - 1)")
    print("=" * 60)

    def bose_einstein(x):
        eps = x[:, 0] + 0.5  # Shift to avoid singularity
        return 1 / (np.exp(eps) - 1 + 1e-8)

    dataset4 = create_dataset(
        f=bose_einstein,
        n_var=1,
        ranges=(0.1, 5),
        train_num=500,
        test_num=100,
        seed=42,
    )

    model4 = MultKAN(width=[1, 1], grid=15, k=3, seed=42)
    model4.fit(dataset4, steps=200, lr=0.02, verbose=False)

    print("\nTop symbolic suggestions:")
    suggestions = model4.suggest_symbolic(0, 0, 0, top_k=5)
    for fn, r2, _ in suggestions:
        print(f"  {fn:15s} R² = {r2:.4f}")

    # =========================================================================
    # Example 5: Two-variable physics function
    # =========================================================================
    print("\n" + "=" * 60)
    print("Example 5: Damped Harmonic Oscillator")
    print("Target: f(t, ω) = exp(-t) * cos(ω*t)")
    print("=" * 60)

    def damped_oscillator(x):
        t = x[:, 0]
        omega = 3.0  # Fixed frequency
        return np.exp(-np.abs(t)) * np.cos(omega * t)

    dataset5 = create_dataset(
        f=damped_oscillator,
        n_var=1,
        ranges=(0, 5),
        train_num=500,
        test_num=100,
        seed=42,
    )

    model5 = MultKAN(width=[1, 1], grid=20, k=3, seed=42)
    model5.fit(dataset5, steps=200, lr=0.02, verbose=False)

    print("\nTop symbolic suggestions:")
    suggestions = model5.suggest_symbolic(0, 0, 0, top_k=5)
    for fn, r2, _ in suggestions:
        print(f"  {fn:15s} R² = {r2:.4f}")

    # Try to fix with damped_cos
    if suggestions[0][1] > 0.9:
        best_fn = suggestions[0][0]
        model5.fix_symbolic(0, 0, 0, best_fn, verbose=True)

        print("\nFormula (Typst):")
        print(f"  ${model5.symbolic_formula_typst(var_names=['t'])}$")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
