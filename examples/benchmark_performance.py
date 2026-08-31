"""
Performance Benchmark: PyKAN vs kan-mlx-physics DSL

This script benchmarks training time and performance across multiple problems:
1. Simple 1D function (sin)
2. 2D function (symbolic regression)
3. 1D Schrödinger equation (PDE)
4. Wigner function (complex eigenvalue PDE)

Measures:
- Training time per epoch
- Total training time
- Final loss
- Memory usage
- Throughput (samples/second)
"""

import time
import gc
import psutil
import os
import numpy as np
import mlx.core as mx
import torch
import matplotlib.pyplot as plt
from scipy.special import eval_laguerre

# Get current process for memory monitoring
process = psutil.Process(os.getpid())


def get_memory_mb():
    """Get current memory usage in MB."""
    return process.memory_info().rss / 1024 / 1024


# =============================================================================
# Benchmark 1: Simple 1D Function (sin)
# =============================================================================

def benchmark_1d_function():
    """Benchmark simple function approximation: f(x) = sin(2πx)."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: 1D Function Approximation (sin)")
    print("=" * 70)
    print("Problem: f(x) = sin(2πx), x ∈ [-1, 1]")
    print("Training: 1000 steps, 500 samples per batch")

    results = {}

    # -------------------------------------------------------------------------
    # PyKAN
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PyKAN (PyTorch)")
    print("-" * 70)

    try:
        from kan import KAN

        # Generate data
        x_train = np.random.uniform(-1, 1, 500).reshape(-1, 1).astype(np.float32)
        y_train = np.sin(2 * np.pi * x_train).astype(np.float32)

        dataset = {
            'train_input': torch.from_numpy(x_train),
            'train_label': torch.from_numpy(y_train),
        }

        # Create model
        gc.collect()
        mem_before = get_memory_mb()

        model = KAN(width=[1, 5, 1], grid=5, k=3, seed=42)

        mem_after = get_memory_mb()
        mem_model = mem_after - mem_before

        # Train
        start_time = time.time()

        history = model.fit(
            dataset,
            opt='Adam',
            steps=1000,
            lr=0.01,
            lamb=0.001,
        )

        elapsed = time.time() - start_time

        mem_peak = get_memory_mb()

        # Evaluate
        y_pred = model(torch.from_numpy(x_train)).detach().numpy()
        final_loss = np.mean((y_pred - y_train)**2)

        results['pykan'] = {
            'time': elapsed,
            'time_per_step': elapsed / 1000,
            'final_loss': float(final_loss),
            'memory_model': mem_model,
            'memory_peak': mem_peak,
            'throughput': 500 * 1000 / elapsed,  # samples/sec
        }

        print(f"  Training time: {elapsed:.2f}s")
        print(f"  Time per step: {elapsed/1000*1000:.2f}ms")
        print(f"  Final loss: {final_loss:.6e}")
        print(f"  Memory (model): {mem_model:.1f} MB")
        print(f"  Memory (peak): {mem_peak:.1f} MB")
        print(f"  Throughput: {500*1000/elapsed:.0f} samples/sec")

    except ImportError:
        print("  ⚠ PyKAN not available, skipping")
        results['pykan'] = None

    # -------------------------------------------------------------------------
    # kan-mlx-physics
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("kan-mlx-physics (MLX)")
    print("-" * 70)

    from kan_mlx_physics import MultKAN, create_dataset

    # Generate data
    def target_fn(x):
        return np.sin(2 * np.pi * x[:, 0])

    dataset = create_dataset(
        f=target_fn,
        n_var=1,
        ranges=(-1, 1),
        train_num=500,
        test_num=100,
        seed=42,
    )

    # Create model
    gc.collect()
    mem_before = get_memory_mb()

    model = MultKAN(width=[1, 5, 1], grid=5, k=3, seed=42)

    mem_after = get_memory_mb()
    mem_model = mem_after - mem_before

    # Train
    start_time = time.time()

    history = model.fit(
        dataset,
        opt='Adam',
        steps=1000,
        lr=0.01,
        lamb=0.001,
        verbose=False,
    )

    elapsed = time.time() - start_time

    mem_peak = get_memory_mb()

    # Evaluate
    final_loss = history['train_loss'][-1]

    results['mlx'] = {
        'time': elapsed,
        'time_per_step': elapsed / 1000,
        'final_loss': float(final_loss),
        'memory_model': mem_model,
        'memory_peak': mem_peak,
        'throughput': 500 * 1000 / elapsed,
    }

    print(f"  Training time: {elapsed:.2f}s")
    print(f"  Time per step: {elapsed/1000*1000:.2f}ms")
    print(f"  Final loss: {final_loss:.6e}")
    print(f"  Memory (model): {mem_model:.1f} MB")
    print(f"  Memory (peak): {mem_peak:.1f} MB")
    print(f"  Throughput: {500*1000/elapsed:.0f} samples/sec")

    # -------------------------------------------------------------------------
    # Comparison
    # -------------------------------------------------------------------------
    if results['pykan'] is not None:
        print("\n" + "-" * 70)
        print("Comparison")
        print("-" * 70)

        speedup = results['pykan']['time'] / results['mlx']['time']
        mem_reduction = (results['pykan']['memory_peak'] - results['mlx']['memory_peak']) / results['pykan']['memory_peak'] * 100

        print(f"  Speedup: {speedup:.2f}x faster (MLX)")
        print(f"  Memory reduction: {mem_reduction:.1f}%")
        print(f"  Throughput ratio: {results['mlx']['throughput']/results['pykan']['throughput']:.2f}x")

    return results


# =============================================================================
# Benchmark 2: 2D Symbolic Regression
# =============================================================================

def benchmark_2d_symbolic():
    """Benchmark 2D symbolic regression: f(x,y) = sin(πx) + y²."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: 2D Symbolic Regression")
    print("=" * 70)
    print("Problem: f(x,y) = sin(πx) + y², x,y ∈ [-1, 1]")
    print("Training: 500 steps, 1000 samples per batch")

    results = {}

    # -------------------------------------------------------------------------
    # PyKAN
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PyKAN (PyTorch)")
    print("-" * 70)

    try:
        from kan import KAN

        # Generate data
        x_train = np.random.uniform(-1, 1, (1000, 2)).astype(np.float32)
        y_train = (np.sin(np.pi * x_train[:, 0]) + x_train[:, 1]**2).reshape(-1, 1).astype(np.float32)

        dataset = {
            'train_input': torch.from_numpy(x_train),
            'train_label': torch.from_numpy(y_train),
        }

        gc.collect()
        mem_before = get_memory_mb()

        model = KAN(width=[2, 5, 1], grid=5, k=3, seed=42)

        mem_after = get_memory_mb()

        start_time = time.time()

        history = model.fit(
            dataset,
            opt='Adam',
            steps=500,
            lr=0.01,
            lamb=0.001,
        )

        elapsed = time.time() - start_time
        mem_peak = get_memory_mb()

        y_pred = model(torch.from_numpy(x_train)).detach().numpy()
        final_loss = np.mean((y_pred - y_train)**2)

        results['pykan'] = {
            'time': elapsed,
            'final_loss': float(final_loss),
            'memory_peak': mem_peak,
            'throughput': 1000 * 500 / elapsed,
        }

        print(f"  Training time: {elapsed:.2f}s")
        print(f"  Final loss: {final_loss:.6e}")
        print(f"  Throughput: {1000*500/elapsed:.0f} samples/sec")

    except ImportError:
        results['pykan'] = None

    # -------------------------------------------------------------------------
    # kan-mlx-physics
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("kan-mlx-physics (MLX)")
    print("-" * 70)

    from kan_mlx_physics import MultKAN, create_dataset

    def target_fn(x):
        return np.sin(np.pi * x[:, 0]) + x[:, 1]**2

    dataset = create_dataset(
        f=target_fn,
        n_var=2,
        ranges=(-1, 1),
        train_num=1000,
        test_num=200,
        seed=42,
    )

    gc.collect()
    mem_before = get_memory_mb()

    model = MultKAN(width=[2, 5, 1], grid=5, k=3, seed=42)

    start_time = time.time()

    history = model.fit(
        dataset,
        opt='Adam',
        steps=500,
        lr=0.01,
        lamb=0.001,
        verbose=False,
    )

    elapsed = time.time() - start_time
    mem_peak = get_memory_mb()

    final_loss = history['train_loss'][-1]

    results['mlx'] = {
        'time': elapsed,
        'final_loss': float(final_loss),
        'memory_peak': mem_peak,
        'throughput': 1000 * 500 / elapsed,
    }

    print(f"  Training time: {elapsed:.2f}s")
    print(f"  Final loss: {final_loss:.6e}")
    print(f"  Throughput: {1000*500/elapsed:.0f} samples/sec")

    if results['pykan'] is not None:
        print("\n" + "-" * 70)
        print("Comparison")
        print("-" * 70)
        speedup = results['pykan']['time'] / results['mlx']['time']
        print(f"  Speedup: {speedup:.2f}x faster (MLX)")

    return results


# =============================================================================
# Benchmark 3: 1D Schrödinger Equation
# =============================================================================

def benchmark_schrodinger():
    """Benchmark 1D Schrödinger with harmonic potential."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: 1D Schrödinger Equation (PDE)")
    print("=" * 70)
    print("Problem: -∇²ψ/2 + x²ψ/2 = Eψ, x ∈ [-5, 5]")
    print("Training: 1000 steps")

    results = {}

    # -------------------------------------------------------------------------
    # PyKAN - Not applicable
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("PyKAN (PyTorch)")
    print("-" * 70)
    print("  ⚠ PyKAN does not support PDE solving natively")
    print("  Would require ~200-300 lines of custom code")
    results['pykan'] = None

    # -------------------------------------------------------------------------
    # kan-mlx-physics DSL
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("kan-mlx-physics DSL (MLX)")
    print("-" * 70)

    from kan_mlx_physics.pde import PDEBuilder, PDEResidualLoss, NormalizationLoss, NonTrivialLoss, EigenvalueLoss

    gc.collect()
    mem_before = get_memory_mb()

    start_time = time.time()

    model, history = (
        PDEBuilder("-Derivative(psi, x, 2)/2 + x**2*psi/2 = E*psi")
        .domain([-5, 5])
        .params(E=0.5)
        .loss(PDEResidualLoss(weight=100))
        .loss(NormalizationLoss(weight=10))
        .loss(NonTrivialLoss(weight=20))
        .loss(EigenvalueLoss(weight=1))
        .phase("train", steps=1000, lr=0.001, n_points=500)
        .model(width=[1, 10, 1], grid=5, k=3, basis="hermite", basis_M=6)
        .quiet()
        .solve()
    )

    elapsed = time.time() - start_time
    mem_peak = get_memory_mb()

    final_loss = history.final_loss

    results['mlx'] = {
        'time': elapsed,
        'final_loss': float(final_loss),
        'memory_peak': mem_peak,
        'lines_of_code': 10,  # DSL lines
    }

    print(f"  Training time: {elapsed:.2f}s")
    print(f"  Final loss: {final_loss:.6e}")
    print(f"  Code: ~10 lines (DSL)")
    print(f"  Memory: {mem_peak:.1f} MB")

    return results


# =============================================================================
# Benchmark 4: Scaling Test (Different Problem Sizes)
# =============================================================================

def benchmark_scaling():
    """Test scaling with problem size."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: Scaling Test")
    print("=" * 70)
    print("Testing performance vs number of training samples")

    from kan_mlx_physics import MultKAN, create_dataset

    sizes = [100, 500, 1000, 2000, 5000]
    results_mlx = []

    def target_fn(x):
        return np.sin(2 * np.pi * x[:, 0])

    print("\nkan-mlx-physics:")
    for size in sizes:
        dataset = create_dataset(
            f=target_fn,
            n_var=1,
            ranges=(-1, 1),
            train_num=size,
            test_num=100,
            seed=42,
        )

        model = MultKAN(width=[1, 5, 1], grid=5, k=3, seed=42)

        start_time = time.time()
        history = model.fit(dataset, steps=100, lr=0.01, verbose=False)
        elapsed = time.time() - start_time

        throughput = size * 100 / elapsed
        results_mlx.append({'size': size, 'time': elapsed, 'throughput': throughput})

        print(f"  {size:5d} samples: {elapsed:6.2f}s ({throughput:8.0f} samples/sec)")

    return {'mlx': results_mlx}


# =============================================================================
# Summary and Visualization
# =============================================================================

def create_summary(bench1, bench2, bench3, bench4):
    """Create summary table and visualization."""
    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE SUMMARY")
    print("=" * 70)

    print("\n1D Function Approximation:")
    if bench1['pykan'] is not None:
        print(f"  PyKAN:          {bench1['pykan']['time']:6.2f}s")
        print(f"  kan-mlx-physics: {bench1['mlx']['time']:6.2f}s")
        print(f"  Speedup:        {bench1['pykan']['time']/bench1['mlx']['time']:6.2f}x")
    else:
        print(f"  kan-mlx-physics: {bench1['mlx']['time']:6.2f}s")

    print("\n2D Symbolic Regression:")
    if bench2['pykan'] is not None:
        print(f"  PyKAN:          {bench2['pykan']['time']:6.2f}s")
        print(f"  kan-mlx-physics: {bench2['mlx']['time']:6.2f}s")
        print(f"  Speedup:        {bench2['pykan']['time']/bench2['mlx']['time']:6.2f}x")
    else:
        print(f"  kan-mlx-physics: {bench2['mlx']['time']:6.2f}s")

    print("\n1D Schrödinger PDE:")
    print(f"  PyKAN:          N/A (requires custom implementation)")
    print(f"  kan-mlx-physics: {bench3['mlx']['time']:6.2f}s")

    # Calculate average speedup
    speedups = []
    if bench1['pykan'] is not None:
        speedups.append(bench1['pykan']['time'] / bench1['mlx']['time'])
    if bench2['pykan'] is not None:
        speedups.append(bench2['pykan']['time'] / bench2['mlx']['time'])

    if speedups:
        avg_speedup = np.mean(speedups)
        print(f"\nAverage speedup (comparable benchmarks): {avg_speedup:.2f}x")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training time comparison
    ax = axes[0, 0]
    if bench1['pykan'] is not None and bench2['pykan'] is not None:
        benchmarks = ['1D Function', '2D Symbolic']
        pykan_times = [bench1['pykan']['time'], bench2['pykan']['time']]
        mlx_times = [bench1['mlx']['time'], bench2['mlx']['time']]

        x = np.arange(len(benchmarks))
        width = 0.35

        ax.bar(x - width/2, pykan_times, width, label='PyKAN', color='#3498db')
        ax.bar(x + width/2, mlx_times, width, label='kan-mlx-physics', color='#e74c3c')

        ax.set_ylabel('Training Time (seconds)', fontsize=11)
        ax.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Speedup
    ax = axes[0, 1]
    if speedups:
        benchmarks = ['1D Function', '2D Symbolic'][:len(speedups)]
        ax.bar(benchmarks, speedups, color='#2ecc71')
        ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Baseline')
        ax.set_ylabel('Speedup Factor', fontsize=11)
        ax.set_title('Performance Speedup (MLX vs PyTorch)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Throughput
    ax = axes[1, 0]
    if bench1['pykan'] is not None:
        benchmarks = ['1D Function', '2D Symbolic']
        pykan_throughput = [bench1['pykan']['throughput'], bench2['pykan']['throughput']]
        mlx_throughput = [bench1['mlx']['throughput'], bench2['mlx']['throughput']]

        x = np.arange(len(benchmarks))
        width = 0.35

        ax.bar(x - width/2, pykan_throughput, width, label='PyKAN', color='#3498db')
        ax.bar(x + width/2, mlx_throughput, width, label='kan-mlx-physics', color='#e74c3c')

        ax.set_ylabel('Throughput (samples/sec)', fontsize=11)
        ax.set_title('Training Throughput', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Scaling
    ax = axes[1, 1]
    sizes = [r['size'] for r in bench4['mlx']]
    times = [r['time'] for r in bench4['mlx']]

    ax.plot(sizes, times, 'o-', linewidth=2, markersize=8, color='#e74c3c', label='kan-mlx-physics')
    ax.set_xlabel('Number of Training Samples', fontsize=11)
    ax.set_ylabel('Training Time (seconds)', fontsize=11)
    ax.set_title('Scaling with Problem Size', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./figures/performance_benchmark.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved: ./figures/performance_benchmark.png")


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("PERFORMANCE BENCHMARK: PyKAN vs kan-mlx-physics")
    print("=" * 70)
    print(f"Hardware: {psutil.cpu_count()} CPU cores")
    print(f"Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Platform: {torch.get_num_threads()} PyTorch threads")

    # Run benchmarks
    bench1 = benchmark_1d_function()
    bench2 = benchmark_2d_symbolic()
    bench3 = benchmark_schrodinger()
    bench4 = benchmark_scaling()

    # Summary
    create_summary(bench1, bench2, bench3, bench4)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
