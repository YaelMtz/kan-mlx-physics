#!/usr/bin/env python3
"""
KAN-MLX-Physics Benchmark Script

Reproducible benchmarks for comparing KAN-MLX-Physics performance.
Run this script to generate performance metrics for your hardware.

Usage:
    python benchmarks/benchmark.py

Results will be printed to console and optionally saved to benchmark_results.json.

Hardware requirements:
    - macOS with Apple Silicon (M1/M2/M3/M4)
    - Python 3.10+
    - MLX 0.21+
"""

import json
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import mlx.core as mx
import numpy as np

# Try to import kan_mlx_physics (handle case where not installed)
try:
    from kan_mlx_physics import MultKAN, create_dataset
    from kan_mlx_physics.functional import (
        functional_forward,
        get_params_list,
        init_adam_state,
        adam_update,
    )
except ImportError:
    print("Error: kan_mlx_physics not installed.")
    print("Run: pip install -e '.' from the project root directory.")
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    # Model configuration
    width: list = None
    grid: int = 5
    k: int = 3

    # Training configuration
    train_steps: int = 2000
    batch_size: int = 1000
    learning_rate: float = 0.01

    # Benchmark configuration
    warmup_steps: int = 10
    n_runs: int = 3

    def __post_init__(self):
        if self.width is None:
            self.width = [2, 5, 1]


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    name: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    n_runs: int


def get_system_info() -> dict:
    """Collect system information for reproducibility."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "mlx_version": mx.__version__,
        "timestamp": datetime.now().isoformat(),
        "machine": platform.machine(),
    }


def benchmark_forward_pass(
    config: BenchmarkConfig,
    n_warmup: int = 10,
    n_runs: int = 100,
) -> BenchmarkResult:
    """Benchmark forward pass throughput."""

    model = MultKAN(width=config.width, grid=config.grid, k=config.k)
    x = mx.random.uniform(shape=(config.batch_size, config.width[0]))

    # Warmup
    for _ in range(n_warmup):
        _ = model(x)
        mx.eval(_)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        y = model(x)
        mx.eval(y)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return BenchmarkResult(
        name="forward_pass",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
        n_runs=n_runs,
    )


def benchmark_gradient_computation(
    config: BenchmarkConfig,
    n_warmup: int = 10,
    n_runs: int = 50,
) -> BenchmarkResult:
    """Benchmark gradient computation."""

    model = MultKAN(width=config.width, grid=config.grid, k=config.k)
    params = get_params_list(model)
    x = mx.random.uniform(shape=(config.batch_size, config.width[0]))
    y_target = mx.random.uniform(shape=(config.batch_size, config.width[-1]))

    def loss_fn(params):
        y_pred = functional_forward(params, x, config.k, model.base_fun)
        return mx.mean((y_pred - y_target) ** 2)

    # Warmup
    for _ in range(n_warmup):
        loss, grads = mx.value_and_grad(loss_fn)(params)
        mx.eval(loss, grads)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        loss, grads = mx.value_and_grad(loss_fn)(params)
        mx.eval(loss, grads)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return BenchmarkResult(
        name="gradient_computation",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
        n_runs=n_runs,
    )


def benchmark_training(
    config: BenchmarkConfig,
    n_runs: int = 3,
) -> BenchmarkResult:
    """Benchmark full training loop."""

    # Create dataset
    def target_fn(x):
        return mx.sin(mx.pi * x[:, 0]) + x[:, 1] ** 2

    dataset = create_dataset(
        f=target_fn,
        n_var=config.width[0],
        train_num=config.batch_size,
        test_num=200,
    )

    times = []
    for run in range(n_runs):
        # Fresh model for each run
        model = MultKAN(width=config.width, grid=config.grid, k=config.k, seed=run)
        params = get_params_list(model)
        m, v, t = init_adam_state(params)

        x_train = dataset["train_input"]
        y_train = dataset["train_label"]

        def loss_fn(params):
            y_pred = functional_forward(params, x_train, config.k, model.base_fun)
            return mx.mean((y_pred - y_train) ** 2)

        # Time training loop
        start = time.perf_counter()
        for step in range(config.train_steps):
            loss, grads = mx.value_and_grad(loss_fn)(params)
            params, m, v, t = adam_update(params, grads, m, v, t, lr=config.learning_rate)

            # Evaluate every 100 steps to ensure computation happens
            if step % 100 == 0:
                mx.eval(params)

        mx.eval(params)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        print(f"  Run {run + 1}/{n_runs}: {elapsed:.2f}s (final loss: {float(loss):.6f})")

    return BenchmarkResult(
        name=f"training_{config.train_steps}_steps",
        mean_time_ms=np.mean(times) * 1000,
        std_time_ms=np.std(times) * 1000,
        min_time_ms=np.min(times) * 1000,
        max_time_ms=np.max(times) * 1000,
        n_runs=n_runs,
    )


def benchmark_symbolic_regression(
    config: BenchmarkConfig,
    n_runs: int = 5,
) -> BenchmarkResult:
    """Benchmark symbolic regression."""

    # Train a simple model first
    def target_fn(x):
        return mx.sin(mx.pi * x[:, 0]) + x[:, 1] ** 2

    dataset = create_dataset(
        f=target_fn,
        n_var=2,
        train_num=1000,
    )

    model = MultKAN(width=[2, 5, 1], grid=5, k=3, seed=42)
    model.fit(dataset, steps=500, lr=0.01, log=0)

    x_sample = dataset["train_input"]

    # Benchmark symbolic regression
    times = []
    for _ in range(n_runs):
        # Reset symbolic state
        for layer in model.symbolic_funs:
            layer.mask = mx.zeros_like(layer.mask)

        start = time.perf_counter()
        model.auto_symbolic(x_sample, r2_threshold=0.9)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return BenchmarkResult(
        name="symbolic_regression",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
        n_runs=n_runs,
    )


def run_all_benchmarks(
    config: Optional[BenchmarkConfig] = None,
    save_results: bool = True,
) -> dict:
    """Run all benchmarks and return results."""

    if config is None:
        config = BenchmarkConfig()

    print("=" * 60)
    print("KAN-MLX-Physics Benchmark Suite")
    print("=" * 60)
    print()

    # System info
    sys_info = get_system_info()
    print("System Information:")
    print(f"  Platform: {sys_info['platform']}")
    print(f"  Processor: {sys_info['processor']}")
    print(f"  Python: {sys_info['python_version']}")
    print(f"  MLX: {sys_info['mlx_version']}")
    print()

    print(f"Benchmark Configuration:")
    print(f"  Model width: {config.width}")
    print(f"  Grid: {config.grid}, k: {config.k}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Training steps: {config.train_steps}")
    print()

    results = {}

    # Forward pass benchmark
    print("-" * 40)
    print("1. Forward Pass Benchmark")
    print("-" * 40)
    result = benchmark_forward_pass(config)
    results["forward_pass"] = {
        "mean_ms": result.mean_time_ms,
        "std_ms": result.std_time_ms,
        "min_ms": result.min_time_ms,
        "max_ms": result.max_time_ms,
    }
    print(f"  Mean: {result.mean_time_ms:.2f} +/- {result.std_time_ms:.2f} ms")
    print()

    # Gradient computation benchmark
    print("-" * 40)
    print("2. Gradient Computation Benchmark")
    print("-" * 40)
    result = benchmark_gradient_computation(config)
    results["gradient_computation"] = {
        "mean_ms": result.mean_time_ms,
        "std_ms": result.std_time_ms,
        "min_ms": result.min_time_ms,
        "max_ms": result.max_time_ms,
    }
    print(f"  Mean: {result.mean_time_ms:.2f} +/- {result.std_time_ms:.2f} ms")
    print()

    # Training benchmark
    print("-" * 40)
    print(f"3. Training Benchmark ({config.train_steps} steps)")
    print("-" * 40)
    result = benchmark_training(config, n_runs=3)
    results["training"] = {
        "mean_s": result.mean_time_ms / 1000,
        "std_s": result.std_time_ms / 1000,
        "min_s": result.min_time_ms / 1000,
        "max_s": result.max_time_ms / 1000,
        "steps": config.train_steps,
    }
    print(f"  Mean: {result.mean_time_ms / 1000:.2f} +/- {result.std_time_ms / 1000:.2f} s")
    print()

    # Symbolic regression benchmark
    print("-" * 40)
    print("4. Symbolic Regression Benchmark")
    print("-" * 40)
    result = benchmark_symbolic_regression(config)
    results["symbolic_regression"] = {
        "mean_ms": result.mean_time_ms,
        "std_ms": result.std_time_ms,
        "min_ms": result.min_time_ms,
        "max_ms": result.max_time_ms,
    }
    print(f"  Mean: {result.mean_time_ms:.2f} +/- {result.std_time_ms:.2f} ms")
    print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Forward pass:          {results['forward_pass']['mean_ms']:.2f} ms")
    print(f"  Gradient computation:  {results['gradient_computation']['mean_ms']:.2f} ms")
    print(f"  Training ({config.train_steps} steps):  {results['training']['mean_s']:.2f} s")
    print(f"  Symbolic regression:   {results['symbolic_regression']['mean_ms']:.2f} ms")
    print()

    # Save results
    if save_results:
        output = {
            "system_info": sys_info,
            "config": {
                "width": config.width,
                "grid": config.grid,
                "k": config.k,
                "batch_size": config.batch_size,
                "train_steps": config.train_steps,
            },
            "results": results,
        }

        output_path = "benchmarks/benchmark_results.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run KAN-MLX-Physics benchmarks")
    parser.add_argument("--steps", type=int, default=2000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--grid", type=int, default=5, help="Spline grid size")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    config = BenchmarkConfig(
        train_steps=args.steps,
        batch_size=args.batch_size,
        grid=args.grid,
    )

    run_all_benchmarks(config, save_results=not args.no_save)
