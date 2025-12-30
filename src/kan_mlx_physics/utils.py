"""Utility functions for MLX-KAN."""

import mlx.core as mx
from typing import Callable, Dict, Tuple, Optional, Union
import numpy as np


def create_dataset(
    f: Callable,
    n_var: int = 2,
    ranges: Union[list, Tuple[float, float]] = (-1.0, 1.0),
    train_num: int = 1000,
    test_num: int = 1000,
    normalize_input: bool = False,
    normalize_label: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, mx.array]:
    """Create a synthetic dataset from a function.

    Args:
        f: Function to generate labels from inputs
        n_var: Number of input variables
        ranges: Input ranges, either (min, max) for all vars or list of (min, max) per var
        train_num: Number of training samples
        test_num: Number of test samples
        normalize_input: Whether to normalize inputs to [-1, 1]
        normalize_label: Whether to normalize labels to [-1, 1]
        seed: Random seed

    Returns:
        Dictionary with train_input, train_label, test_input, test_label
    """
    if seed is not None:
        mx.random.seed(seed)
        np.random.seed(seed)

    # Handle ranges
    if isinstance(ranges, tuple):
        ranges = [ranges] * n_var

    # Generate random inputs
    def generate_inputs(n_samples: int) -> mx.array:
        inputs = []
        for i in range(n_var):
            low, high = ranges[i]
            x = mx.random.uniform(low=low, high=high, shape=(n_samples, 1))
            inputs.append(x)
        return mx.concatenate(inputs, axis=1)

    train_input = generate_inputs(train_num)
    test_input = generate_inputs(test_num)

    # Generate labels
    # Convert to numpy for arbitrary function evaluation, then back to mlx
    train_input_np = np.array(train_input)
    test_input_np = np.array(test_input)

    train_label_np = f(train_input_np)
    test_label_np = f(test_input_np)

    # Ensure proper shape
    if train_label_np.ndim == 1:
        train_label_np = train_label_np[:, np.newaxis]
    if test_label_np.ndim == 1:
        test_label_np = test_label_np[:, np.newaxis]

    train_label = mx.array(train_label_np)
    test_label = mx.array(test_label_np)

    # Normalize if requested
    if normalize_input:
        input_mean = mx.mean(train_input, axis=0, keepdims=True)
        input_std = mx.std(train_input, axis=0, keepdims=True) + 1e-8
        train_input = (train_input - input_mean) / input_std
        test_input = (test_input - input_mean) / input_std

    if normalize_label:
        label_mean = mx.mean(train_label, axis=0, keepdims=True)
        label_std = mx.std(train_label, axis=0, keepdims=True) + 1e-8
        train_label = (train_label - label_mean) / label_std
        test_label = (test_label - label_mean) / label_std

    return {
        "train_input": train_input,
        "train_label": train_label,
        "test_input": test_input,
        "test_label": test_label,
    }


def create_dataset_from_data(
    inputs: Union[np.ndarray, mx.array],
    labels: Union[np.ndarray, mx.array],
    train_ratio: float = 0.8,
    normalize_input: bool = False,
    normalize_label: bool = False,
    seed: Optional[int] = None,
) -> Dict[str, mx.array]:
    """Create a dataset from existing data.

    Args:
        inputs: Input data of shape (n_samples, n_features)
        labels: Label data of shape (n_samples, n_outputs) or (n_samples,)
        train_ratio: Fraction of data for training
        normalize_input: Whether to normalize inputs
        normalize_label: Whether to normalize labels
        seed: Random seed for shuffling

    Returns:
        Dictionary with train_input, train_label, test_input, test_label
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert to numpy for shuffling
    if isinstance(inputs, mx.array):
        inputs = np.array(inputs)
    if isinstance(labels, mx.array):
        labels = np.array(labels)

    # Ensure proper shape
    if labels.ndim == 1:
        labels = labels[:, np.newaxis]

    # Shuffle
    n_samples = inputs.shape[0]
    indices = np.random.permutation(n_samples)
    inputs = inputs[indices]
    labels = labels[indices]

    # Split
    n_train = int(n_samples * train_ratio)
    train_input = mx.array(inputs[:n_train])
    train_label = mx.array(labels[:n_train])
    test_input = mx.array(inputs[n_train:])
    test_label = mx.array(labels[n_train:])

    # Normalize if requested
    if normalize_input:
        input_mean = mx.mean(train_input, axis=0, keepdims=True)
        input_std = mx.std(train_input, axis=0, keepdims=True) + 1e-8
        train_input = (train_input - input_mean) / input_std
        test_input = (test_input - input_mean) / input_std

    if normalize_label:
        label_mean = mx.mean(train_label, axis=0, keepdims=True)
        label_std = mx.std(train_label, axis=0, keepdims=True) + 1e-8
        train_label = (train_label - label_mean) / label_std
        test_label = (test_label - label_mean) / label_std

    return {
        "train_input": train_input,
        "train_label": train_label,
        "test_input": test_input,
        "test_label": test_label,
    }


# Standard test functions
def f_sin(x: np.ndarray) -> np.ndarray:
    """f(x) = sin(pi * x)"""
    return np.sin(np.pi * x[:, 0])


def f_exp(x: np.ndarray) -> np.ndarray:
    """f(x) = exp(sum(x))"""
    return np.exp(np.sum(x, axis=1))


def f_gaussian(x: np.ndarray) -> np.ndarray:
    """f(x) = exp(-sum(x^2))"""
    return np.exp(-np.sum(x ** 2, axis=1))


def f_polynomial(x: np.ndarray) -> np.ndarray:
    """f(x, y) = x^2 + y^2"""
    return x[:, 0] ** 2 + x[:, 1] ** 2


def f_mixed(x: np.ndarray) -> np.ndarray:
    """f(x, y) = sin(pi * x) + y^2"""
    return np.sin(np.pi * x[:, 0]) + x[:, 1] ** 2


def f_feynman_1(x: np.ndarray) -> np.ndarray:
    """Feynman equation: kinetic energy E = 0.5 * m * v^2"""
    m, v = x[:, 0], x[:, 1]
    return 0.5 * m * v ** 2


def f_feynman_2(x: np.ndarray) -> np.ndarray:
    """Feynman equation: gravitational force F = G * m1 * m2 / r^2"""
    m1, m2, r = x[:, 0], x[:, 1], x[:, 2]
    G = 1.0  # Normalized
    return G * m1 * m2 / (r ** 2 + 1e-8)
