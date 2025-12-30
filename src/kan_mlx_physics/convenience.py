"""High-level convenience functions for common KAN workflows.

These functions reduce boilerplate for common tasks:
- quick_fit: One-liner to create, train, and get a model
- auto_formula: Extract symbolic formula from trained model
- visualize: Unified visualization interface

Example:
    >>> from kan_mlx_physics import quick_fit, auto_formula
    >>> model, history = quick_fit(
    ...     f=lambda x: mx.sin(mx.pi * x[:, 0]) + x[:, 1]**2,
    ...     n_var=2,
    ... )
    >>> formula = auto_formula(model, history['x_sample'], ['x', 'y'])
"""

import mlx.core as mx
from typing import Optional, Dict, Any, Callable, List, Tuple, Union


def quick_fit(
    f: Callable[[mx.array], mx.array],
    n_var: int,
    width: Optional[List[int]] = None,
    preset: Optional[str] = None,
    steps: int = 500,
    train_num: int = 1000,
    test_num: Optional[int] = None,
    domain: Union[Tuple[float, float], List[Tuple[float, float]]] = (-1, 1),
    learning_rate: float = 0.01,
    regularization: float = 0.01,
    verbose: bool = True,
    **model_kwargs
) -> Tuple["MultKAN", Dict[str, Any]]:
    """One-liner to create and train a KAN model.

    Creates a dataset, builds a model, trains it, and returns both
    the trained model and training history.

    Args:
        f: Target function f(x) -> y where x is (batch, n_var)
        n_var: Number of input variables
        width: Network architecture. If None, uses [n_var, 5*n_var, 1]
        preset: Physics preset name (overrides basis settings in model_kwargs)
        steps: Number of training steps
        train_num: Number of training samples
        test_num: Number of test samples (default: train_num // 5)
        domain: Input domain as (min, max) or list of per-variable ranges
        learning_rate: Learning rate (alias: lr)
        regularization: Regularization strength (alias: lamb)
        verbose: Print training progress
        **model_kwargs: Additional MultKAN arguments (grid, basis, etc.)

    Returns:
        Tuple of (trained_model, training_history_dict)
        History dict includes 'x_sample' key with sample input for formula extraction

    Example:
        >>> model, history = quick_fit(
        ...     f=lambda x: mx.sin(mx.pi * x[:, 0]) + x[:, 1]**2,
        ...     n_var=2,
        ...     steps=500,
        ... )

        >>> # With physics preset
        >>> model, history = quick_fit(
        ...     f=harmonic_oscillator,
        ...     n_var=1,
        ...     preset="quantum_oscillator",
        ... )
    """
    from .utils import create_dataset
    from .presets import from_preset
    from .multkan import MultKAN

    # Default architecture
    if width is None:
        width = [n_var, 5 * n_var, 1]

    # Default test size
    if test_num is None:
        test_num = max(train_num // 5, 100)

    # Create dataset
    dataset = create_dataset(
        f=f,
        n_var=n_var,
        train_num=train_num,
        test_num=test_num,
        ranges=domain,
    )

    # Create model
    if preset:
        model = from_preset(preset, width=width, **model_kwargs)
    else:
        model = MultKAN(width=width, **model_kwargs)

    # Train
    # Use large log_freq when not verbose to avoid division by zero
    log_freq = 50 if verbose else steps + 1
    history = model.fit(
        dataset,
        steps=steps,
        learning_rate=learning_rate,
        regularization=regularization,
        log_freq=log_freq,
        verbose=verbose,
    )

    # Add sample data to history for formula extraction
    history['x_sample'] = dataset['train_input']

    return model, history


def auto_formula(
    model: "MultKAN",
    x_sample: mx.array,
    var_names: Optional[List[str]] = None,
    threshold: float = 0.95,
    format: str = "unicode",
    verbose: bool = False,
) -> str:
    """Extract symbolic formula from a trained model.

    Runs a forward pass to cache activations, performs automatic
    symbolic regression, and returns the formula in the requested format.

    Args:
        model: Trained MultKAN model
        x_sample: Sample input data for activation caching
        var_names: Variable names. If None, uses ['x_0', 'x_1', ...]
        threshold: R^2 threshold for symbolic fitting (0.0 to 1.0)
        format: Output format: "unicode", "latex", "typst", "sympy"
        verbose: Print symbolic fitting details

    Returns:
        Formula string in the requested format

    Example:
        >>> formula = auto_formula(model, dataset['train_input'], ['x', 'y'])
        >>> print(formula)  # sin(3.14*x) + y^2

        >>> latex = auto_formula(model, x, ['x', 'y'], format="latex")
    """
    # Run forward pass to cache activations
    _ = model(x_sample)

    # Auto-detect symbolic functions
    model.auto_symbolic(x_sample, r2_threshold=threshold, verbose=verbose)

    # Generate variable names if not provided
    if var_names is None:
        n_var = x_sample.shape[1]
        var_names = [f"x_{i}" for i in range(n_var)]

    # Return in requested format
    if format == "unicode" or format == "text":
        return model.symbolic_formula(var_names, verbose=False)
    elif format == "latex":
        return model.symbolic_formula_latex(var_names)
    elif format == "typst":
        return model.symbolic_formula_typst(var_names)
    elif format == "sympy":
        expr = model.to_sympy(var_names)
        return str(expr)
    else:
        valid_formats = ["unicode", "text", "latex", "typst", "sympy"]
        raise ValueError(f"Unknown format: {format}. Use one of: {valid_formats}")


def visualize(
    model: "MultKAN",
    what: str = "network",
    x: Optional[mx.array] = None,
    **kwargs
) -> None:
    """Unified visualization interface for KAN models.

    Provides a single entry point for all visualization needs.

    Args:
        model: MultKAN model to visualize
        what: What to visualize:
            - "network": Full network diagram with activation functions (default)
            - "activations": Activation functions for a specific layer
            - "history": Training history plot
        x: Input data (required for "activations", optional for "network")
        **kwargs: Additional arguments passed to underlying plot function:
            - For "network": folder, beta, scale, title, in_vars, out_vars
            - For "activations": layer (layer index, default 0), folder
            - For "history": history (required), metric ("loss" or "reg")

    Example:
        >>> # Network diagram
        >>> visualize(model, "network", folder="./figs", title="My KAN")

        >>> # Activation functions
        >>> visualize(model, "activations", x=data, layer=0)

        >>> # Training history
        >>> visualize(model, "history", history=training_history)
    """
    if what == "network":
        model.plot(**kwargs)

    elif what == "activations":
        if x is None:
            raise ValueError("x is required for activation visualization")
        layer_idx = kwargs.pop("layer", 0)
        model.plot_activations(layer_idx=layer_idx, x=x, **kwargs)

    elif what == "history":
        history = kwargs.pop("history", None)
        if history is None:
            raise ValueError("history dict is required for history visualization")

        # Import matplotlib
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for visualization. Install with: pip install matplotlib")

        metric = kwargs.pop("metric", "loss")

        fig, ax = plt.subplots(figsize=(8, 5))

        if metric == "loss":
            if "train_loss" in history:
                ax.plot(history["train_loss"], label="Train Loss")
            if "test_loss" in history and len(history["test_loss"]) > 0:
                ax.plot(history["test_loss"], label="Test Loss")
            ax.set_ylabel("Loss (MSE)")
        elif metric == "reg":
            if "reg_loss" in history:
                ax.plot(history["reg_loss"], label="Regularization")
            ax.set_ylabel("Regularization Loss")
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'loss' or 'reg'")

        ax.set_xlabel("Step")
        ax.set_title(kwargs.get("title", "Training History"))
        ax.legend()
        ax.grid(True, alpha=0.3)

        if kwargs.get("save", False):
            folder = kwargs.get("folder", "./figures")
            import os
            os.makedirs(folder, exist_ok=True)
            fig.savefig(f"{folder}/training_history.png", dpi=150, bbox_inches="tight")

        if kwargs.get("display", True):
            plt.show()
        else:
            plt.close()

    else:
        valid_options = ["network", "activations", "history"]
        raise ValueError(f"Unknown visualization: {what}. Use one of: {valid_options}")


def fit_and_extract(
    f: Callable[[mx.array], mx.array],
    n_var: int,
    var_names: Optional[List[str]] = None,
    **kwargs
) -> Tuple["MultKAN", str]:
    """Convenience function to train a model and extract the formula.

    Combines quick_fit and auto_formula into a single call.

    Args:
        f: Target function
        n_var: Number of input variables
        var_names: Names for variables in formula
        **kwargs: Arguments passed to quick_fit

    Returns:
        Tuple of (trained_model, formula_string)

    Example:
        >>> model, formula = fit_and_extract(
        ...     f=lambda x: mx.sin(mx.pi * x[:, 0]),
        ...     n_var=1,
        ...     var_names=['x'],
        ... )
        >>> print(formula)  # sin(3.14*x)
    """
    model, history = quick_fit(f, n_var, **kwargs)
    formula = auto_formula(model, history['x_sample'], var_names)
    return model, formula
