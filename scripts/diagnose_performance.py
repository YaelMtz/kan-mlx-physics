"""Performance Diagnostic Script

Runs 3 trials of a simple PDE problem with detailed timing breakdowns.
Identifies bottlenecks causing 119% variance in training time.

Usage:
    python scripts/diagnose_performance.py

Expected output:
    Identifies which operation (grid updates, eval sync, forward/backward)
    has the highest variance, confirming the root cause.
"""

import time
import contextlib
import numpy as np
import mlx.core as mx
from kan_mlx_physics.pde import (
    PDEBuilder,
    PDEResidualLoss,
    NormalizationLoss,
)


class TimingContext:
    """Context manager for timing operations with proper MLX synchronization"""

    def __init__(self, name: str, timings: dict):
        self.name = name
        self.timings = timings
        self.start = None

    def __enter__(self):
        # Force MLX synchronization before timing
        mx.eval(mx.array([0.0]))
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        # Force MLX synchronization after timing
        mx.eval(mx.array([0.0]))
        elapsed = time.perf_counter() - self.start

        if self.name not in self.timings:
            self.timings[self.name] = []
        self.timings[self.name].append(elapsed)


def run_diagnostic_trial(trial_num: int, seed: int = 42) -> dict:
    """Run one diagnostic trial with detailed timing

    Args:
        trial_num: Trial number for display
        seed: Random seed for reproducibility

    Returns:
        Dictionary of timing measurements
    """
    print(f"\n{'='*60}")
    print(f"Trial {trial_num} (seed={seed})")
    print(f"{'='*60}")

    timings = {}

    with TimingContext("total", timings):
        # Build problem (simple harmonic oscillator)
        with TimingContext("problem_setup", timings):
            equation = "-Derivative(psi, x, 2)/2 + x**2*psi/2 = E*psi"
            builder = (
                PDEBuilder(equation)
                .domain([-5, 5])
                .params(E=0.5)
                .loss(PDEResidualLoss(weight=100))
                .loss(NormalizationLoss(weight=10))
            )

        # Phase 1: Train without grid update
        print("Phase 1: Training (500 steps, no grid update)...")
        with TimingContext("phase1_total", timings):
            model, history = (
                builder
                .phase("train", steps=500, lr=0.001, n_points=500, log_freq=10000)
                .model(width=[1, 10, 1], grid=5, k=3, seed=seed)
                .solve(verbose=False)
            )

        print(f"  Final loss: {history.final_loss:.6e}")

        # Phase 2: Test grid update
        print("\nPhase 2: Testing grid update...")
        with TimingContext("grid_update", timings):
            # Sample points for grid update
            x_grid = mx.linspace(-5, 5, 1000).reshape(-1, 1)
            # This should trigger the expensive operation
            model.update_grid_from_samples(x_grid)

        # Phase 3: Forward/backward timing (post grid update)
        print("\nPhase 3: Testing forward/backward after grid update...")
        with TimingContext("forward_after_grid", timings):
            x_test = mx.random.uniform(-5, 5, (500, 1))
            for _ in range(10):
                y = model(x_test)
                mx.eval(y)

        print(f"  10 forward passes complete")

    return timings


def analyze_timings(all_trials: list[dict]):
    """Analyze timing results across trials and identify bottlenecks

    Args:
        all_trials: List of timing dictionaries from each trial
    """
    print(f"\n{'='*60}")
    print("DIAGNOSTIC RESULTS")
    print(f"{'='*60}\n")

    # Compute statistics for each timing category
    categories = set()
    for trial in all_trials:
        categories.update(trial.keys())

    results = []
    for cat in sorted(categories):
        values = []
        for trial in all_trials:
            if cat in trial:
                values.extend(trial[cat])

        if not values:
            continue

        mean = np.mean(values)
        std = np.std(values)
        variance = (std / mean * 100) if mean > 0 else 0

        results.append({
            'category': cat,
            'mean': mean,
            'std': std,
            'variance': variance,
            'min': np.min(values),
            'max': np.max(values)
        })

    # Sort by variance (highest first)
    results.sort(key=lambda x: x['variance'], reverse=True)

    print(f"{'Operation':<25} {'Mean':>10} {'Std':>10} {'Variance':>10} {'Range':>15}")
    print("-" * 80)

    for r in results:
        variance_marker = "⚠️ " if r['variance'] > 50 else "   "
        print(f"{r['category']:<25} {r['mean']:>8.3f}s {r['std']:>8.3f}s "
              f"{r['variance']:>8.1f}% {variance_marker} "
              f"{r['min']:.2f}-{r['max']:.2f}s")

    # Identify dominant hypothesis
    print(f"\n{'='*60}")
    print("HYPOTHESIS RANKING")
    print(f"{'='*60}\n")

    # Find highest variance operations
    if results:
        top_issue = results[0]

        if top_issue['variance'] > 50:
            print(f"🔴 HIGH VARIANCE DETECTED: {top_issue['category']}")
            print(f"   Variance: {top_issue['variance']:.1f}%")
            print(f"   Range: {top_issue['min']:.2f}s - {top_issue['max']:.2f}s")

            if 'grid' in top_issue['category'].lower():
                print(f"\n✓ H1 CONFIRMED: Grid updates causing instability")
                print(f"   Recommendation: Add mx.eval() synchronization in grid update")
            elif 'phase' in top_issue['category'].lower() or 'total' in top_issue['category'].lower():
                print(f"\n✓ H2 LIKELY: Missing mx.eval() in training loop")
                print(f"   Recommendation: Add step-level synchronization")
            else:
                print(f"\n⚠️  Unexpected bottleneck - manual investigation needed")
        else:
            print(f"✓ LOW VARIANCE: All operations stable (<50%)")
            print(f"   The issue may be environmental (thermal throttling, etc.)")

    # Calculate total variance
    total_times = []
    for trial in all_trials:
        if 'total' in trial:
            total_times.extend(trial['total'])

    if total_times:
        total_mean = np.mean(total_times)
        total_std = np.std(total_times)
        total_variance = (total_std / total_mean * 100)

        print(f"\nOVERALL VARIANCE: {total_variance:.1f}%")
        if total_variance > 10:
            print(f"⚠️  Above 10% threshold - fix needed")
        else:
            print(f"✓ Below 10% threshold - acceptable")


def main():
    """Run diagnostic trials"""
    print("="*60)
    print("MLX PERFORMANCE DIAGNOSTIC")
    print("="*60)
    print("\nRunning 3 trials to identify bottleneck...")
    print("Each trial uses different random seed")

    n_trials = 3
    all_trials = []

    for i in range(n_trials):
        seed = 42 + i
        trial_timings = run_diagnostic_trial(i + 1, seed=seed)
        all_trials.append(trial_timings)

        # Brief summary
        total_time = trial_timings.get('total', [0])[0]
        print(f"  Trial {i+1} total time: {total_time:.2f}s")

    # Analyze results
    analyze_timings(all_trials)

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Check which timing category has highest variance (>50%)")
    print("2. Apply targeted fix based on dominant hypothesis:")
    print("   - H1 (Grid updates): Add mx.eval() in kan_layer.py")
    print("   - H2 (Missing sync): Add mx.eval() in trainer.py")
    print("3. Re-run this diagnostic to verify variance drops below 10%")
    print("\n")


if __name__ == "__main__":
    main()
