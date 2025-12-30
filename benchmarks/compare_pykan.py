#!/usr/bin/env python3
"""
PyKAN vs KAN-MLX-Physics: Side-by-Side Benchmark

Comprehensive performance comparison between PyKAN and KAN-MLX-Physics including:
- Forward pass throughput
- Gradient computation
- Full training loop
- Symbolic regression
- Memory usage
- Scaling analysis

Usage:
    # Install benchmark dependencies first
    pip install -e ".[benchmark]"

    # Run full comparison
    python benchmarks/compare_pykan.py

    # Options
    python benchmarks/compare_pykan.py --steps 2000 --batch-size 1000
    python benchmarks/compare_pykan.py --scaling        # Include scaling analysis
    python benchmarks/compare_pykan.py --mlx-only       # Skip PyKAN (if not installed)
    python benchmarks/compare_pykan.py --output results.json

Hardware requirements:
    - macOS with Apple Silicon (M1/M2/M3/M4)
    - Python 3.10+
    - MLX 0.21+
    - PyKAN 0.2+ (optional, for comparison)
"""

import gc
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Any

import mlx.core as mx
import numpy as np

# Try to import psutil for memory profiling
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Memory profiling disabled.")
    print("Install with: pip install psutil")

# Try to import KAN-MLX-Physics
try:
    from kan_mlx_physics import MultKAN, create_dataset
    from kan_mlx_physics.functional import (
        functional_forward,
        get_params_list,
        init_adam_state,
        adam_update,
    )
    HAS_MLX_KAN = True
except ImportError:
    HAS_MLX_KAN = False
    print("Error: kan_mlx_physics not installed.")
    print("Run: pip install -e '.' from the project root directory.")
    sys.exit(1)

# Try to import PyKAN
try:
    import torch
    from kan import KAN
    HAS_PYKAN = True
    PYKAN_VERSION = "0.2.x"  # PyKAN doesn't expose __version__
    TORCH_VERSION = torch.__version__
except ImportError:
    HAS_PYKAN = False
    PYKAN_VERSION = None
    TORCH_VERSION = None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    width: List[int] = field(default_factory=lambda: [2, 5, 1])
    grid: int = 5
    k: int = 3
    train_steps: int = 2000
    batch_size: int = 1000
    learning_rate: float = 0.01
    warmup_runs: int = 10
    n_runs: int = 3
    seed: int = 42


@dataclass
class BenchmarkResult:
    """Results from a single benchmark."""
    framework: str
    benchmark: str
    model_size: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    memory_mb: float
    final_loss: Optional[float]
    n_runs: int
    extra: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Utility Functions
# =============================================================================

def get_input_dim(width: List) -> int:
    """Extract input dimension from width, handling extended format."""
    first = width[0]
    if isinstance(first, (list, tuple)):
        return first[0]  # Extended format: [[n_sum, n_mult], ...]
    return first


def get_system_info() -> Dict[str, str]:
    """Collect system information for reproducibility."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "mlx_version": mx.__version__,
        "timestamp": datetime.now().isoformat(),
    }
    if HAS_PYKAN:
        info["pykan_version"] = PYKAN_VERSION
        info["torch_version"] = TORCH_VERSION
    return info


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    if not HAS_PSUTIL:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def measure_peak_memory(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """Measure peak memory usage during function execution."""
    if not HAS_PSUTIL:
        result = func(*args, **kwargs)
        return result, 0.0

    gc.collect()
    baseline = get_memory_mb()

    result = func(*args, **kwargs)

    peak = get_memory_mb()
    return result, max(0, peak - baseline)


def format_model_size(width: List[int]) -> str:
    """Format model width as string."""
    return f"[{', '.join(map(str, width))}]"


def count_parameters(width: List[int], grid: int, k: int) -> int:
    """Estimate number of parameters in a KAN model."""
    total = 0
    for i in range(len(width) - 1):
        # Spline coefficients: in_dim * out_dim * (grid + k)
        total += width[i] * width[i + 1] * (grid + k)
        # Scale parameters
        total += width[i] * width[i + 1] * 2
    return total


# =============================================================================
# Target Functions
# =============================================================================

def target_function_mlx(x: mx.array) -> mx.array:
    """Standard benchmark function: f(x, y) = sin(pi*x) + y^2"""
    return mx.sin(mx.pi * x[:, 0]) + x[:, 1] ** 2


def target_function_torch(x: "torch.Tensor") -> "torch.Tensor":
    """Standard benchmark function for PyTorch: f(x, y) = sin(pi*x) + y^2"""
    import torch
    return torch.sin(torch.pi * x[:, 0]) + x[:, 1] ** 2


# =============================================================================
# MLX Benchmarks
# =============================================================================

def benchmark_mlx_forward(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark MLX forward pass throughput."""
    model = MultKAN(width=config.width, grid=config.grid, k=config.k, seed=config.seed)
    input_dim = get_input_dim(config.width)
    x = mx.random.uniform(shape=(config.batch_size, input_dim))

    # Warmup
    for _ in range(config.warmup_runs):
        y = model(x)
        mx.eval(y)

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        y = model(x)
        mx.eval(y)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    memory = get_memory_mb()

    return BenchmarkResult(
        framework="mlx",
        benchmark="forward_pass",
        model_size=format_model_size(config.width),
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
        memory_mb=memory,
        final_loss=None,
        n_runs=100,
    )


def benchmark_mlx_gradients(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark MLX gradient computation."""
    model = MultKAN(width=config.width, grid=config.grid, k=config.k, seed=config.seed)
    params = get_params_list(model)
    input_dim = get_input_dim(config.width)
    x = mx.random.uniform(shape=(config.batch_size, input_dim))
    y_target = target_function_mlx(x).reshape(-1, 1)

    def loss_fn(params):
        y_pred = functional_forward(params, x, config.k, model.base_fun)
        return mx.mean((y_pred - y_target) ** 2)

    # Warmup
    for _ in range(config.warmup_runs):
        loss, grads = mx.value_and_grad(loss_fn)(params)
        mx.eval(loss, grads)

    # Benchmark
    times = []
    for _ in range(50):
        start = time.perf_counter()
        loss, grads = mx.value_and_grad(loss_fn)(params)
        mx.eval(loss, grads)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    memory = get_memory_mb()

    return BenchmarkResult(
        framework="mlx",
        benchmark="gradient_computation",
        model_size=format_model_size(config.width),
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
        memory_mb=memory,
        final_loss=float(loss),
        n_runs=50,
    )


def benchmark_mlx_training(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark MLX full training loop."""
    input_dim = get_input_dim(config.width)
    dataset = create_dataset(
        f=target_function_mlx,
        n_var=input_dim,
        train_num=config.batch_size,
        test_num=200,
    )

    times = []
    final_losses = []

    for run in range(config.n_runs):
        model = MultKAN(width=config.width, grid=config.grid, k=config.k, seed=config.seed + run)

        gc.collect()
        start_mem = get_memory_mb()

        start = time.perf_counter()
        history = model.fit(
            dataset,
            steps=config.train_steps,
            lr=config.learning_rate,
            lamb=0.01,
            log_freq=config.train_steps + 1,  # Suppress logging
            verbose=False,
        )
        mx.eval(model.parameters())
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        final_losses.append(history['train_loss'][-1])
        print(f"  MLX Run {run + 1}/{config.n_runs}: {elapsed:.2f}s (loss: {final_losses[-1]:.6f})")

    peak_mem = get_memory_mb() - start_mem

    return BenchmarkResult(
        framework="mlx",
        benchmark=f"training_{config.train_steps}_steps",
        model_size=format_model_size(config.width),
        mean_time_ms=np.mean(times) * 1000,
        std_time_ms=np.std(times) * 1000,
        min_time_ms=np.min(times) * 1000,
        max_time_ms=np.max(times) * 1000,
        memory_mb=peak_mem,
        final_loss=np.mean(final_losses),
        n_runs=config.n_runs,
        extra={"steps": config.train_steps},
    )


def benchmark_mlx_symbolic(config: BenchmarkConfig) -> BenchmarkResult:
    """Benchmark MLX symbolic regression."""
    # Train a model first
    dataset = create_dataset(
        f=target_function_mlx,
        n_var=2,
        train_num=1000,
    )

    model = MultKAN(width=[2, 5, 1], grid=5, k=3, seed=config.seed)
    model.fit(dataset, steps=500, lr=0.01, log_freq=1000, verbose=False)

    x_sample = dataset["train_input"]

    # Benchmark symbolic regression
    times = []
    for _ in range(5):
        # Reset symbolic state
        for layer in model.symbolic_funs:
            layer._symbolic_mask = mx.zeros_like(layer._symbolic_mask)

        start = time.perf_counter()
        model.auto_symbolic(x_sample, r2_threshold=0.9, verbose=False)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    memory = get_memory_mb()

    return BenchmarkResult(
        framework="mlx",
        benchmark="symbolic_regression",
        model_size="[2, 5, 1]",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
        memory_mb=memory,
        final_loss=None,
        n_runs=5,
    )


# =============================================================================
# PyKAN Benchmarks
# =============================================================================

def benchmark_pykan_forward(config: BenchmarkConfig) -> Optional[BenchmarkResult]:
    """Benchmark PyKAN forward pass throughput."""
    if not HAS_PYKAN:
        return None

    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = KAN(width=config.width, grid=config.grid, k=config.k, seed=config.seed, device=device)
    input_dim = get_input_dim(config.width)
    x = torch.randn(config.batch_size, input_dim, device=device)

    # Warmup
    for _ in range(config.warmup_runs):
        with torch.no_grad():
            y = model(x)
        if device.type == "mps":
            torch.mps.synchronize()

    # Benchmark
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            y = model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    memory = get_memory_mb()

    return BenchmarkResult(
        framework="pykan",
        benchmark="forward_pass",
        model_size=format_model_size(config.width),
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
        memory_mb=memory,
        final_loss=None,
        n_runs=100,
        extra={"device": str(device)},
    )


def benchmark_pykan_gradients(config: BenchmarkConfig) -> Optional[BenchmarkResult]:
    """Benchmark PyKAN gradient computation."""
    if not HAS_PYKAN:
        return None

    import torch

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = KAN(width=config.width, grid=config.grid, k=config.k, seed=config.seed, device=device)
    input_dim = get_input_dim(config.width)
    x = torch.randn(config.batch_size, input_dim, device=device, requires_grad=True)
    y_target = target_function_torch(x.detach()).unsqueeze(1)

    # Warmup
    for _ in range(config.warmup_runs):
        y_pred = model(x)
        loss = torch.mean((y_pred - y_target) ** 2)
        loss.backward()
        model.zero_grad()
        if device.type == "mps":
            torch.mps.synchronize()

    # Benchmark
    times = []
    for _ in range(50):
        start = time.perf_counter()
        y_pred = model(x)
        loss = torch.mean((y_pred - y_target) ** 2)
        loss.backward()
        model.zero_grad()
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    memory = get_memory_mb()

    return BenchmarkResult(
        framework="pykan",
        benchmark="gradient_computation",
        model_size=format_model_size(config.width),
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
        memory_mb=memory,
        final_loss=float(loss.item()),
        n_runs=50,
        extra={"device": str(device)},
    )


def benchmark_pykan_training(config: BenchmarkConfig) -> Optional[BenchmarkResult]:
    """Benchmark PyKAN full training loop."""
    if not HAS_PYKAN:
        return None

    import torch

    # PyKAN has issues with MPS for training (lstsq not supported), use CPU
    device = torch.device("cpu")
    input_dim = get_input_dim(config.width)

    # Create dataset compatible with PyKAN
    x_train = torch.randn(config.batch_size, input_dim, device=device)
    y_train = target_function_torch(x_train).unsqueeze(1)

    dataset = {
        "train_input": x_train,
        "train_label": y_train,
        "test_input": torch.randn(200, input_dim, device=device),
    }
    dataset["test_label"] = target_function_torch(dataset["test_input"]).unsqueeze(1)

    times = []
    final_losses = []

    for run in range(config.n_runs):
        model = KAN(width=config.width, grid=config.grid, k=config.k, seed=config.seed + run, device=device)

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        start_mem = get_memory_mb()

        start = time.perf_counter()
        results = model.fit(
            dataset,
            opt="Adam",
            steps=config.train_steps,
            lr=config.learning_rate,
            lamb=0.01,
            log=config.train_steps + 1,  # Log at end only (suppress during training)
        )
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        times.append(elapsed)
        final_loss = results["train_loss"][-1] if "train_loss" in results else 0.0
        final_losses.append(final_loss)
        print(f"  PyKAN Run {run + 1}/{config.n_runs}: {elapsed:.2f}s (loss: {final_loss:.6f})")

    peak_mem = get_memory_mb() - start_mem

    return BenchmarkResult(
        framework="pykan",
        benchmark=f"training_{config.train_steps}_steps",
        model_size=format_model_size(config.width),
        mean_time_ms=np.mean(times) * 1000,
        std_time_ms=np.std(times) * 1000,
        min_time_ms=np.min(times) * 1000,
        max_time_ms=np.max(times) * 1000,
        memory_mb=peak_mem,
        final_loss=np.mean(final_losses),
        n_runs=config.n_runs,
        extra={"steps": config.train_steps, "device": str(device)},
    )


def benchmark_pykan_symbolic(config: BenchmarkConfig) -> Optional[BenchmarkResult]:
    """Benchmark PyKAN symbolic regression."""
    if not HAS_PYKAN:
        return None

    import torch

    # PyKAN has issues with MPS (lstsq not supported), use CPU
    device = torch.device("cpu")

    # Create dataset and train model
    x_train = torch.randn(1000, 2, device=device)
    y_train = target_function_torch(x_train).unsqueeze(1)
    x_test = torch.randn(200, 2, device=device)
    y_test = target_function_torch(x_test).unsqueeze(1)

    dataset = {
        "train_input": x_train,
        "train_label": y_train,
        "test_input": x_test,
        "test_label": y_test,
    }

    model = KAN(width=[2, 5, 1], grid=5, k=3, seed=config.seed, device=device)
    model.fit(dataset, opt="Adam", steps=500, lr=0.01, log=501)

    # Benchmark symbolic regression
    times = []
    for _ in range(5):
        start = time.perf_counter()
        try:
            model.auto_symbolic()
        except Exception:
            pass  # PyKAN symbolic can be finicky
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    memory = get_memory_mb()

    return BenchmarkResult(
        framework="pykan",
        benchmark="symbolic_regression",
        model_size="[2, 5, 1]",
        mean_time_ms=np.mean(times),
        std_time_ms=np.std(times),
        min_time_ms=np.min(times),
        max_time_ms=np.max(times),
        memory_mb=memory,
        final_loss=None,
        n_runs=5,
        extra={"device": str(device)},
    )


# =============================================================================
# Scaling Analysis
# =============================================================================

def run_scaling_analysis(base_config: BenchmarkConfig, mlx_only: bool = False) -> List[BenchmarkResult]:
    """Run scaling analysis across different model sizes."""
    model_sizes = [
        [2, 5, 1],        # Small
        [2, 10, 10, 1],   # Medium
        [2, 20, 20, 1],   # Large
    ]

    results = []

    for width in model_sizes:
        config = BenchmarkConfig(
            width=width,
            grid=base_config.grid,
            k=base_config.k,
            train_steps=base_config.train_steps,
            batch_size=base_config.batch_size,
            learning_rate=base_config.learning_rate,
            n_runs=base_config.n_runs,
            seed=base_config.seed,
        )

        params = count_parameters(width, config.grid, config.k)
        print(f"\n--- Model: {format_model_size(width)} ({params} params) ---")

        # MLX benchmark
        mlx_result = benchmark_mlx_training(config)
        mlx_result.extra["params"] = params
        results.append(mlx_result)

        # PyKAN benchmark
        if not mlx_only and HAS_PYKAN:
            pykan_result = benchmark_pykan_training(config)
            if pykan_result:
                pykan_result.extra["params"] = params
                results.append(pykan_result)

    return results


# =============================================================================
# Output Formatting
# =============================================================================

def print_comparison_table(mlx_results: Dict[str, BenchmarkResult],
                           pykan_results: Dict[str, BenchmarkResult]) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("Performance Comparison: KAN-MLX-Physics vs PyKAN")
    print("=" * 70)

    headers = ["Benchmark", "KAN-MLX-Physics", "PyKAN (MPS)", "Speedup"]
    print(f"\n| {headers[0]:<25} | {headers[1]:<17} | {headers[2]:<14} | {headers[3]:<8} |")
    print(f"|{'-' * 27}|{'-' * 19}|{'-' * 16}|{'-' * 10}|")

    benchmarks = [
        ("forward_pass", "Forward Pass", "ms"),
        ("gradient_computation", "Gradient Computation", "ms"),
        (f"training_{2000}_steps", "Training (2000 steps)", "s"),
        ("symbolic_regression", "Symbolic Regression", "ms"),
    ]

    for key, name, unit in benchmarks:
        mlx = mlx_results.get(key)
        pykan = pykan_results.get(key)

        if mlx:
            if unit == "s":
                mlx_str = f"{mlx.mean_time_ms / 1000:.1f} s"
            else:
                mlx_str = f"{mlx.mean_time_ms:.1f} ms"
        else:
            mlx_str = "N/A"

        if pykan:
            if unit == "s":
                pykan_str = f"{pykan.mean_time_ms / 1000:.1f} s"
            else:
                pykan_str = f"{pykan.mean_time_ms:.1f} ms"
        else:
            pykan_str = "N/A"

        if mlx and pykan and mlx.mean_time_ms > 0:
            speedup = pykan.mean_time_ms / mlx.mean_time_ms
            speedup_str = f"**{speedup:.1f}x**"
        else:
            speedup_str = "N/A"

        print(f"| {name:<25} | {mlx_str:<17} | {pykan_str:<14} | {speedup_str:<8} |")

    # Memory comparison
    print(f"\n{'Memory Usage:':<30}")
    print(f"| {'Framework':<20} | {'Peak Memory':<15} |")
    print(f"|{'-' * 22}|{'-' * 17}|")

    training_key = f"training_{2000}_steps"
    if training_key in mlx_results:
        print(f"| {'KAN-MLX-Physics':<20} | {mlx_results[training_key].memory_mb:.0f} MB{' ' * 8} |")
    if training_key in pykan_results:
        print(f"| {'PyKAN (MPS)':<20} | {pykan_results[training_key].memory_mb:.0f} MB{' ' * 8} |")


def print_scaling_table(results: List[BenchmarkResult]) -> None:
    """Print scaling analysis table."""
    print("\n" + "-" * 60)
    print("Scaling Analysis (Training Time)")
    print("-" * 60)

    # Group by model size
    mlx_by_size = {}
    pykan_by_size = {}

    for r in results:
        if r.framework == "mlx":
            mlx_by_size[r.model_size] = r
        else:
            pykan_by_size[r.model_size] = r

    print(f"\n| {'Model Size':<18} | {'Params':<8} | {'MLX':<10} | {'PyKAN':<10} | {'Speedup':<8} |")
    print(f"|{'-' * 20}|{'-' * 10}|{'-' * 12}|{'-' * 12}|{'-' * 10}|")

    for size in sorted(mlx_by_size.keys()):
        mlx = mlx_by_size.get(size)
        pykan = pykan_by_size.get(size)

        params = mlx.extra.get("params", 0) if mlx else 0
        mlx_time = f"{mlx.mean_time_ms / 1000:.1f}s" if mlx else "N/A"
        pykan_time = f"{pykan.mean_time_ms / 1000:.1f}s" if pykan else "N/A"

        if mlx and pykan and mlx.mean_time_ms > 0:
            speedup = f"{pykan.mean_time_ms / mlx.mean_time_ms:.1f}x"
        else:
            speedup = "N/A"

        print(f"| {size:<18} | {params:<8} | {mlx_time:<10} | {pykan_time:<10} | {speedup:<8} |")


def save_results_json(results: Dict[str, Any], filepath: str) -> None:
    """Save results to JSON file."""
    # Convert BenchmarkResult objects to dicts
    output = {}
    for key, value in results.items():
        if isinstance(value, BenchmarkResult):
            output[key] = asdict(value)
        elif isinstance(value, list):
            output[key] = [asdict(r) if isinstance(r, BenchmarkResult) else r for r in value]
        elif isinstance(value, dict):
            output[key] = {
                k: asdict(v) if isinstance(v, BenchmarkResult) else v
                for k, v in value.items()
            }
        else:
            output[key] = value

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {filepath}")


# =============================================================================
# Main Entry Point
# =============================================================================

def run_all_benchmarks(
    config: BenchmarkConfig,
    include_scaling: bool = False,
    mlx_only: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run all benchmarks and return results."""

    print("=" * 70)
    print("PyKAN vs KAN-MLX-Physics Benchmark Suite")
    print("=" * 70)
    print()

    # System info
    sys_info = get_system_info()
    print("System Information:")
    print(f"  Platform: {sys_info['platform']}")
    print(f"  Processor: {sys_info['processor']}")
    print(f"  Python: {sys_info['python_version']}")
    print(f"  MLX: {sys_info['mlx_version']}")
    if HAS_PYKAN:
        print(f"  PyKAN: {sys_info.get('pykan_version', 'N/A')}")
        print(f"  PyTorch: {sys_info.get('torch_version', 'N/A')}")
    else:
        print("  PyKAN: Not installed (MLX-only mode)")
    print()

    print(f"Benchmark Configuration:")
    print(f"  Model width: {config.width}")
    print(f"  Grid: {config.grid}, k: {config.k}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Training steps: {config.train_steps}")
    print(f"  Runs per benchmark: {config.n_runs}")
    print()

    mlx_results = {}
    pykan_results = {}

    # Forward pass
    print("-" * 50)
    print("1. Forward Pass Benchmark")
    print("-" * 50)
    mlx_results["forward_pass"] = benchmark_mlx_forward(config)
    print(f"  MLX: {mlx_results['forward_pass'].mean_time_ms:.2f} ms")

    if not mlx_only and HAS_PYKAN:
        result = benchmark_pykan_forward(config)
        if result:
            pykan_results["forward_pass"] = result
            print(f"  PyKAN: {result.mean_time_ms:.2f} ms")
    print()

    # Gradient computation
    print("-" * 50)
    print("2. Gradient Computation Benchmark")
    print("-" * 50)
    mlx_results["gradient_computation"] = benchmark_mlx_gradients(config)
    print(f"  MLX: {mlx_results['gradient_computation'].mean_time_ms:.2f} ms")

    if not mlx_only and HAS_PYKAN:
        result = benchmark_pykan_gradients(config)
        if result:
            pykan_results["gradient_computation"] = result
            print(f"  PyKAN: {result.mean_time_ms:.2f} ms")
    print()

    # Training
    print("-" * 50)
    print(f"3. Training Benchmark ({config.train_steps} steps)")
    print("-" * 50)
    training_key = f"training_{config.train_steps}_steps"
    mlx_results[training_key] = benchmark_mlx_training(config)
    print(f"  MLX Mean: {mlx_results[training_key].mean_time_ms / 1000:.2f}s")

    if not mlx_only and HAS_PYKAN:
        result = benchmark_pykan_training(config)
        if result:
            pykan_results[training_key] = result
            print(f"  PyKAN Mean: {result.mean_time_ms / 1000:.2f}s")
    print()

    # Symbolic regression
    print("-" * 50)
    print("4. Symbolic Regression Benchmark")
    print("-" * 50)
    mlx_results["symbolic_regression"] = benchmark_mlx_symbolic(config)
    print(f"  MLX: {mlx_results['symbolic_regression'].mean_time_ms:.2f} ms")

    if not mlx_only and HAS_PYKAN:
        result = benchmark_pykan_symbolic(config)
        if result:
            pykan_results["symbolic_regression"] = result
            print(f"  PyKAN: {result.mean_time_ms:.2f} ms")
    print()

    # Scaling analysis
    scaling_results = []
    if include_scaling:
        print("-" * 50)
        print("5. Scaling Analysis")
        print("-" * 50)
        scaling_results = run_scaling_analysis(config, mlx_only=mlx_only)

    # Print comparison tables
    print_comparison_table(mlx_results, pykan_results)

    if scaling_results:
        print_scaling_table(scaling_results)

    # Compile all results
    all_results = {
        "system_info": sys_info,
        "config": asdict(config),
        "mlx_results": mlx_results,
        "pykan_results": pykan_results,
        "scaling_results": scaling_results,
    }

    # Save to JSON
    if output_path:
        save_results_json(all_results, output_path)

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PyKAN vs KAN-MLX-Physics Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python benchmarks/compare_pykan.py
    python benchmarks/compare_pykan.py --steps 2000 --batch-size 1000
    python benchmarks/compare_pykan.py --scaling
    python benchmarks/compare_pykan.py --mlx-only
    python benchmarks/compare_pykan.py --output results.json
        """
    )

    parser.add_argument("--steps", type=int, default=2000, help="Training steps (default: 2000)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size (default: 1000)")
    parser.add_argument("--grid", type=int, default=5, help="Spline grid size (default: 5)")
    parser.add_argument("--k", type=int, default=3, help="Spline order (default: 3)")
    parser.add_argument("--runs", type=int, default=3, help="Runs per benchmark (default: 3)")
    parser.add_argument("--scaling", action="store_true", help="Include scaling analysis")
    parser.add_argument("--mlx-only", action="store_true", help="Run MLX benchmarks only (skip PyKAN)")
    parser.add_argument("--output", type=str, default="benchmarks/comparison_results.json",
                        help="Output JSON file path")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to JSON")

    args = parser.parse_args()

    config = BenchmarkConfig(
        width=[2, 5, 1],
        grid=args.grid,
        k=args.k,
        train_steps=args.steps,
        batch_size=args.batch_size,
        n_runs=args.runs,
    )

    output_path = None if args.no_save else args.output

    run_all_benchmarks(
        config,
        include_scaling=args.scaling,
        mlx_only=args.mlx_only,
        output_path=output_path,
    )
