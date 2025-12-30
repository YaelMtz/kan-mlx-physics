"""Thesis Example: Deformation Quantization in Cosmology with KANs.

This demonstrates solving the central equations of your thesis:
1. Wheeler-DeWitt equation (quantum cosmology)
2. Deformed Wheeler-DeWitt with Moyal star product
3. Comparison of classical vs quantum corrections

Usage:
    python examples/thesis_cosmology.py
"""

import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt

from kan_mlx_physics.pde import (
    solve,
    WheelerDeWitt,
    DeformedWheelerDeWitt,
    Schrodinger,
    DeformedSchrodinger,
    solve_problem,
    SolverConfig,
)
from kan_mlx_physics import register_physics_symbolic


def main():
    print("=" * 60)
    print("Deformation Quantization in Cosmology with KANs")
    print("=" * 60)

    # Register physics symbolic functions
    register_physics_symbolic()

    # =========================================================================
    # 1. Standard Wheeler-DeWitt
    # =========================================================================
    print("\n1. Standard Wheeler-DeWitt Equation")
    print("-" * 40)
    print("Ĥ Ψ = 0  where Ĥ = -ℏ² ∂²/∂a² + U(a)")

    wdw_problem = WheelerDeWitt()

    wdw_config = SolverConfig(
        width=[1, 20, 20, 1],
        steps=200,
        n_interior=500,
        n_boundary=100,
        log_freq=100,
        sampling='uniform',
    )

    wdw_model, wdw_history = solve_problem(wdw_problem, wdw_config)
    print(f"\nFinal loss: {wdw_history['loss'][-1]:.4e}")

    # =========================================================================
    # 2. Deformed Wheeler-DeWitt (Moyal)
    # =========================================================================
    print("\n2. Deformed Wheeler-DeWitt (Moyal Star Product)")
    print("-" * 40)
    print("Ĥ ⋆_θ Ψ = 0  with θ = 0.05")

    deformed_problem = DeformedWheelerDeWitt(theta=0.05)

    deformed_config = SolverConfig(
        width=[2, 20, 20, 1],  # 2D phase space input
        steps=200,
        n_interior=600,
        n_boundary=150,
        log_freq=100,
        sampling='uniform',
    )

    deformed_model, deformed_history = solve_problem(deformed_problem, deformed_config)
    print(f"\nFinal loss: {deformed_history['loss'][-1]:.4e}")

    # =========================================================================
    # 3. Compare Solutions
    # =========================================================================
    print("\n3. Comparing Solutions")
    print("-" * 40)

    # Evaluate standard WDW
    a_vals = mx.linspace(0.5, 4.5, 50).reshape(-1, 1)
    psi_standard = wdw_model(a_vals)

    # Evaluate deformed WDW at π_a = 0 slice
    a_vals_2d = mx.concatenate([
        a_vals,
        mx.zeros((50, 1))  # π_a = 0
    ], axis=1)
    psi_deformed = deformed_model(a_vals_2d)

    # Convert to numpy for plotting
    a_np = np.array(a_vals[:, 0])
    psi_std_np = np.array(psi_standard[:, 0])
    psi_def_np = np.array(psi_deformed[:, 0])

    # Normalize
    psi_std_np = psi_std_np / (np.max(np.abs(psi_std_np)) + 1e-8)
    psi_def_np = psi_def_np / (np.max(np.abs(psi_def_np)) + 1e-8)

    print(f"  Ψ_standard(a=2) = {psi_std_np[20]:.4f}")
    print(f"  Ψ_deformed(a=2, π=0) = {psi_def_np[20]:.4f}")
    print(f"  Relative difference: {abs(psi_std_np[20] - psi_def_np[20]):.4f}")

    # =========================================================================
    # 4. DSL Interface Demo
    # =========================================================================
    print("\n4. DSL Interface Demo")
    print("-" * 40)
    print("Type equations naturally:")

    # Solve via typed equation
    print("\n  solve('Ĥ Ψ = 0', domain=[0.1, 5])")
    dsl_model, _ = solve("wheeler-dewitt", domain=[0.1, 5], steps=50, verbose=False)
    print("  ✓ Solved!")

    print("\n  solve('Ĥ ⋆_θ Ψ = 0', params={'θ': 0.1})")
    # This would need the deformed template
    print("  (Use DeformedWheelerDeWitt for Moyal deformation)")

    # =========================================================================
    # 5. Plot Results
    # =========================================================================
    print("\n5. Saving plot to figures/thesis_cosmology.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: Wave functions
    axes[0].plot(a_np, psi_std_np, 'b-', label='Standard WDW', linewidth=2)
    axes[0].plot(a_np, psi_def_np, 'r--', label=f'Deformed (θ=0.05)', linewidth=2)
    axes[0].set_xlabel('Scale factor a')
    axes[0].set_ylabel('Ψ(a) [normalized]')
    axes[0].set_title('Wheeler-DeWitt Wave Functions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Training history
    axes[1].semilogy(wdw_history['step'], wdw_history['loss'], 'b-', label='Standard')
    axes[1].semilogy(deformed_history['step'], deformed_history['loss'], 'r-', label='Deformed')
    axes[1].set_xlabel('Training Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Convergence')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Phase space (deformed only)
    a_grid = np.linspace(0.5, 4.5, 30)
    pi_grid = np.linspace(-3, 3, 30)
    A, PI = np.meshgrid(a_grid, pi_grid)
    points = np.stack([A.flatten(), PI.flatten()], axis=1)
    psi_2d = deformed_model(mx.array(points.astype(np.float32)))
    psi_2d = np.array(psi_2d).reshape(30, 30)

    im = axes[2].contourf(A, PI, psi_2d, levels=20, cmap='RdBu_r')
    axes[2].set_xlabel('Scale factor a')
    axes[2].set_ylabel('Conjugate momentum π_a')
    axes[2].set_title('Deformed Ψ in Phase Space')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    plt.savefig('figures/thesis_cosmology.png', dpi=150)
    print("  ✓ Saved!")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
