"""Compare Autodiff vs Finite Differences: Speed to Analytical Solution

This script trains the Wigner problem until it converges to the analytical
eigenvalue E₀ = 0.5 (ground state) and compares:
1. Time to convergence
2. Number of steps required
3. Final accuracy achieved

Tests both derivative methods:
- autodiff (default, accurate)
- finite_diff (legacy, less accurate)

Usage:
    python scripts/compare_autodiff_vs_finite_diff.py
"""

import time
import numpy as np
import mlx.core as mx
from kan_mlx_physics.pde import (
    PDEBuilder,
    PDEResidualLoss,
    NormalizationLoss,
    NonTrivialLoss,
    DecayLoss,
    SmoothnessLoss,
    AnchorLoss,
    EigenvalueLoss,
    RegularizationLoss,
)


DOMAIN = 3.0
HBAR = 1.0
E_ANALYTICAL = 0.5  # Ground state eigenvalue
CONVERGENCE_THRESHOLD = 0.05  # 5% error threshold


def wigner_rayleigh(trainer, x, params):
    """Rayleigh quotient for Wigner eigenvalue"""
    hbar = params.get("hbar", HBAR)
    hbar2_4 = hbar**2 / 4.0
    r2 = mx.squeeze(x, axis=-1)
    W = trainer.u(x)
    dW_dr2 = trainer.du(x)
    grad_W_squared = 4.0 * r2 * (dW_dr2**2)
    numerator = mx.mean(r2 * (W**2) + hbar2_4 * grad_W_squared)
    denominator = 2.0 * mx.mean(W**2) + 1e-8
    return numerator / denominator


def extract_eigenvalue(history):
    """Extract final eigenvalue from training history"""
    if hasattr(history, 'final_eigenvalue') and history.final_eigenvalue is not None:
        return history.final_eigenvalue
    elif hasattr(history, 'phases') and history.phases:
        last_phase = history.phases[-1]
        if hasattr(last_phase, 'eigenvalues') and last_phase.eigenvalues:
            return last_phase.eigenvalues[-1]
    return None


def train_until_convergence(method: str, max_phases: int = 10) -> dict:
    """Train Wigner problem until convergence to analytical solution.

    Args:
        method: "autodiff" or "finite_diff"
        max_phases: Maximum number of training phases

    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"Testing: {method.upper()}")
    print(f"{'='*70}")
    print(f"Target eigenvalue: E₀ = {E_ANALYTICAL}")
    print(f"Convergence threshold: {CONVERGENCE_THRESHOLD*100}% error\n")

    equation = "(r2 - 2*E)*W - (hbar**2/4)*(4*r2*Derivative(W, r2, 2) + 4*Derivative(W, r2)) = 0"

    total_steps = 0
    total_time = 0.0
    converged = False
    phase_num = 0

    # Initial training phase
    print(f"Phase 1: Initial training (2000 steps)...")
    start_time = time.time()

    builder = (
        PDEBuilder(equation)
        .derivative_method(method)  # Set derivative method
        .params(hbar=HBAR)
        .domain([0, 2 * DOMAIN**2])
        .loss(PDEResidualLoss(weight=10.0, normalize=True))
        .loss(NormalizationLoss(weight=10.0))
        .loss(NonTrivialLoss(weight=20.0))
        .loss(DecayLoss(weight=10.0, threshold_ratio=0.8))
        .loss(SmoothnessLoss(weight=0.5))
        .loss(AnchorLoss(x0=mx.array([[0.0]]), target=1.0/(np.pi*HBAR), weight=10.0))
        .loss(EigenvalueLoss(weight=1.0, method="rayleigh", rayleigh_fn=wigner_rayleigh))
        .loss(RegularizationLoss(weight=0.1, norm="l1"))
        .phase("initial", steps=2000, lr=0.001, n_points=1000, log_freq=10000)
        .model(
            width=[1, 2, 1],
            grid=5,
            k=3,
            basis="laguerre",
            basis_M=8,
            basis_kwargs={"alpha": 0.0, "weighted": True},
            grid_range=(0, 2 * DOMAIN**2),
            seed=42,
        )
    )

    model, history = builder.solve(verbose=False)

    phase_time = time.time() - start_time
    total_steps += 2000
    total_time += phase_time
    phase_num += 1

    E_current = extract_eigenvalue(history)
    if E_current is None:
        print("Warning: Could not extract eigenvalue")
        E_current = 10.0  # Large value to indicate non-convergence

    error = abs(E_current - E_ANALYTICAL) / E_ANALYTICAL

    print(f"  Time: {phase_time:.2f}s")
    print(f"  Eigenvalue: E = {E_current:.6f}")
    print(f"  Error: {error*100:.2f}%")
    print(f"  Final loss: {history.final_loss:.3e}")

    if error < CONVERGENCE_THRESHOLD:
        converged = True
        print(f"  ✓ CONVERGED!")

    # Continue training until convergence
    while not converged and phase_num < max_phases:
        phase_num += 1
        print(f"\nPhase {phase_num}: Refinement (1000 steps)...")

        start_time = time.time()

        # Add refinement phase
        builder = (
            PDEBuilder(equation)
            .derivative_method(method)  # Keep same method
            .params(hbar=HBAR)
            .domain([0, 2 * DOMAIN**2])
            .loss(PDEResidualLoss(weight=10.0, normalize=True))
            .loss(NormalizationLoss(weight=10.0))
            .loss(NonTrivialLoss(weight=20.0))
            .loss(DecayLoss(weight=10.0, threshold_ratio=0.8))
            .loss(SmoothnessLoss(weight=0.5))
            .loss(AnchorLoss(x0=mx.array([[0.0]]), target=1.0/(np.pi*HBAR), weight=10.0))
            .loss(EigenvalueLoss(weight=1.0, method="rayleigh", rayleigh_fn=wigner_rayleigh))
            .loss(RegularizationLoss(weight=0.1, norm="l1"))
            .phase("refine", steps=1000, lr=0.0003, n_points=2000,
                   grid_update_before=True, log_freq=10000)
            .model(
                width=[1, 2, 1],
                grid=5,
                k=3,
                basis="laguerre",
                basis_M=8,
                basis_kwargs={"alpha": 0.0, "weighted": True},
                grid_range=(0, 2 * DOMAIN**2),
                seed=42,
            )
        )

        model, history = builder.solve(verbose=False)

        phase_time = time.time() - start_time
        total_steps += 1000
        total_time += phase_time

        E_current = extract_eigenvalue(history)
        if E_current is None:
            E_current = 10.0

        error = abs(E_current - E_ANALYTICAL) / E_ANALYTICAL

        print(f"  Time: {phase_time:.2f}s")
        print(f"  Eigenvalue: E = {E_current:.6f}")
        print(f"  Error: {error*100:.2f}%")
        print(f"  Final loss: {history.final_loss:.3e}")

        if error < CONVERGENCE_THRESHOLD:
            converged = True
            print(f"  ✓ CONVERGED!")
            break

    # Summary
    print(f"\n{'-'*70}")
    if converged:
        print(f"✓ Converged in {phase_num} phases ({total_steps} steps)")
    else:
        print(f"✗ Did not converge after {phase_num} phases ({total_steps} steps)")
    print(f"Total time: {total_time:.2f}s")
    print(f"Final eigenvalue: E = {E_current:.6f} (target: {E_ANALYTICAL:.6f})")
    print(f"Final error: {error*100:.2f}%")

    return {
        'method': method,
        'converged': converged,
        'total_steps': total_steps,
        'total_time': total_time,
        'phases': phase_num,
        'final_eigenvalue': E_current,
        'final_error': error,
        'final_loss': history.final_loss if history else None,
    }


def main():
    """Run comparison between autodiff and finite differences"""
    print("="*70)
    print("AUTODIFF VS FINITE DIFFERENCES: CONVERGENCE SPEED COMPARISON")
    print("="*70)
    print(f"\nWigner Ground State Eigenvalue Problem")
    print(f"Target: E₀ = {E_ANALYTICAL}")
    print(f"Convergence criterion: Error < {CONVERGENCE_THRESHOLD*100}%")
    print(f"\nTraining until convergence...")

    # Test autodiff
    results_autodiff = train_until_convergence("autodiff", max_phases=10)

    # Test finite differences
    results_finite = train_until_convergence("finite_diff", max_phases=10)

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}\n")

    print(f"{'Metric':<30} {'Autodiff':<20} {'Finite Diff':<20} {'Winner'}")
    print(f"{'-'*70}")

    # Convergence
    autodiff_conv = "✓ Yes" if results_autodiff['converged'] else "✗ No"
    finite_conv = "✓ Yes" if results_finite['converged'] else "✗ No"
    conv_winner = ""
    if results_autodiff['converged'] and not results_finite['converged']:
        conv_winner = "Autodiff ✓"
    elif results_finite['converged'] and not results_autodiff['converged']:
        conv_winner = "Finite Diff ✓"
    elif results_autodiff['converged'] and results_finite['converged']:
        conv_winner = "Both ✓"
    print(f"{'Converged:':<30} {autodiff_conv:<20} {finite_conv:<20} {conv_winner}")

    # Steps to convergence
    steps_winner = ""
    if results_autodiff['converged'] and results_finite['converged']:
        if results_autodiff['total_steps'] < results_finite['total_steps']:
            steps_winner = "Autodiff ✓"
            speedup = results_finite['total_steps'] / results_autodiff['total_steps']
        else:
            steps_winner = "Finite Diff ✓"
            speedup = results_autodiff['total_steps'] / results_finite['total_steps']
    print(f"{'Steps:':<30} {results_autodiff['total_steps']:<20} {results_finite['total_steps']:<20} {steps_winner}")

    # Time to convergence
    time_winner = ""
    if results_autodiff['converged'] and results_finite['converged']:
        if results_autodiff['total_time'] < results_finite['total_time']:
            time_winner = "Autodiff ✓"
            time_speedup = results_finite['total_time'] / results_autodiff['total_time']
        else:
            time_winner = "Finite Diff ✓"
            time_speedup = results_autodiff['total_time'] / results_finite['total_time']
    autodiff_time_str = f"{results_autodiff['total_time']:.2f}s"
    finite_time_str = f"{results_finite['total_time']:.2f}s"
    print(f"{'Time:':<30} {autodiff_time_str:<20} {finite_time_str:<20} {time_winner}")

    # Final error
    error_winner = ""
    if results_autodiff['final_error'] < results_finite['final_error']:
        error_winner = "Autodiff ✓"
    else:
        error_winner = "Finite Diff ✓"
    autodiff_error_str = f"{results_autodiff['final_error']*100:.2f}%"
    finite_error_str = f"{results_finite['final_error']*100:.2f}%"
    print(f"{'Final Error:':<30} {autodiff_error_str:<20} {finite_error_str:<20} {error_winner}")

    # Final eigenvalue
    autodiff_eig_str = f"{results_autodiff['final_eigenvalue']:.6f}"
    finite_eig_str = f"{results_finite['final_eigenvalue']:.6f}"
    print(f"{'Final Eigenvalue:':<30} {autodiff_eig_str:<20} {finite_eig_str:<20}")
    analytical_str = f"{E_ANALYTICAL:.6f}"
    print(f"{'Analytical:':<30} {analytical_str:<20} {analytical_str:<20}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    if results_autodiff['converged'] and results_finite['converged']:
        print(f"✓ Both methods converged to the analytical solution")
        print(f"\nSpeed Comparison:")
        if results_autodiff['total_steps'] < results_finite['total_steps']:
            print(f"  - Autodiff converged {speedup:.1f}x faster in steps")
        else:
            print(f"  - Finite diff converged {speedup:.1f}x faster in steps")

        if results_autodiff['total_time'] < results_finite['total_time']:
            print(f"  - Autodiff converged {time_speedup:.1f}x faster in wall-clock time")
        else:
            print(f"  - Finite diff converged {time_speedup:.1f}x faster in wall-clock time")

        print(f"\nAccuracy:")
        if results_autodiff['final_error'] < results_finite['final_error']:
            accuracy_ratio = results_finite['final_error'] / results_autodiff['final_error']
            print(f"  - Autodiff {accuracy_ratio:.1f}x more accurate")
        else:
            accuracy_ratio = results_autodiff['final_error'] / results_finite['final_error']
            print(f"  - Finite diff {accuracy_ratio:.1f}x more accurate")

    elif results_autodiff['converged']:
        print(f"✓ Autodiff converged, finite diff did not")
        print(f"  - Autodiff: {results_autodiff['total_steps']} steps, {results_autodiff['total_time']:.2f}s")
        print(f"  - Finite diff: Failed to converge in {results_finite['total_steps']} steps")

    elif results_finite['converged']:
        print(f"✓ Finite diff converged, autodiff did not")
        print(f"  - Finite diff: {results_finite['total_steps']} steps, {results_finite['total_time']:.2f}s")
        print(f"  - Autodiff: Failed to converge in {results_autodiff['total_steps']} steps")

    else:
        print(f"✗ Neither method converged within {max(results_autodiff['phases'], results_finite['phases'])} phases")

    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}\n")

    if results_autodiff['converged'] or results_autodiff['final_error'] < results_finite['final_error']:
        print("✓ Use AUTODIFF (default)")
        print("  - More accurate derivatives")
        print("  - Better convergence properties")
        print("  - No float32 cancellation issues")
    else:
        print("⚠️  Results suggest finite differences, but autodiff is still recommended")
        print("  - This problem may need more training phases")
        print("  - Autodiff is mathematically superior for derivatives")

    print()


if __name__ == "__main__":
    main()
